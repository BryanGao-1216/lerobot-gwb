#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.models.gemma.configuration_gemma import GemmaConfig

import lerobot.policies.actionmem.modeling_actionmem as modeling_actionmem
from lerobot.configs import NormalizationMode
from lerobot.policies.actionmem.action_vqvae import (
    VQVLALikeDecoder,
    VQVLALikeEncoder,
    _load_checkpoint,
    load_action_vqvae_q0_decoder,
    load_action_vqvae_q0_encoder,
)
from lerobot.policies.actionmem.configuration_actionmem import ActionMemConfig
from lerobot.policies.actionmem.modeling_actionmem import (
    ActionMemPolicy,
    ActionMemPytorch,
    _configure_action_vqvae_flow_normalization,
    _masked_action_token_cross_entropy,
    _restore_vqvla_oxe_actions,
    _select_action_token,
)
from lerobot.policies.pi_gemma import PiGemmaModel
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def test_flow_source_normalization_uses_runtime_dataset_stats_and_persists_them():
    config = type(
        "Config",
        (),
        {
            "normalization_mapping": {"ACTION": NormalizationMode.MEAN_STD},
            "action_vqvae_input_q01": None,
            "action_vqvae_input_q99": None,
            "action_vqvae_input_mask": None,
            "action_vqvae_flow_mean": [100.0, 100.0],
            "action_vqvae_flow_std": [100.0, 100.0],
        },
    )()
    dataset_stats = {
        ACTION: {
            "q01": torch.tensor([-1.0, 0.0]),
            "q99": torch.tensor([1.0, 1.0]),
            "mask": torch.tensor([True, False]),
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([3.0, 4.0]),
        }
    }

    _configure_action_vqvae_flow_normalization(config, dataset_stats)

    assert config.action_vqvae_input_q01 == [-1.0, 0.0]
    assert config.action_vqvae_input_q99 == [1.0, 1.0]
    assert config.action_vqvae_input_mask == [True, False]
    assert config.action_vqvae_flow_mean == [1.0, 2.0]
    assert config.action_vqvae_flow_std == [3.0, 4.0]


def test_flow_source_normalization_reuses_saved_stats_without_dataset_metadata():
    config = type(
        "Config",
        (),
        {
            "normalization_mapping": {"ACTION": NormalizationMode.MEAN_STD},
            "action_vqvae_input_q01": [-1.0],
            "action_vqvae_input_q99": [1.0],
            "action_vqvae_input_mask": [True],
            "action_vqvae_flow_mean": [1.0],
            "action_vqvae_flow_std": [2.0],
        },
    )()

    _configure_action_vqvae_flow_normalization(config, dataset_stats=None)

    assert config.action_vqvae_flow_mean == [1.0]
    assert config.action_vqvae_flow_std == [2.0]


def test_flow_source_normalization_requires_stats_for_mean_std():
    config = type(
        "Config",
        (),
        {
            "normalization_mapping": {"ACTION": NormalizationMode.MEAN_STD},
            "action_vqvae_input_q01": [-1.0],
            "action_vqvae_input_q99": [1.0],
            "action_vqvae_input_mask": [True],
            "action_vqvae_flow_mean": None,
            "action_vqvae_flow_std": None,
        },
    )()

    with pytest.raises(ValueError, match="mean/std are unavailable"):
        _configure_action_vqvae_flow_normalization(config, dataset_stats=None)


def test_restore_vqvla_actions_inverts_bounds_but_preserves_gripper():
    normalized = torch.tensor([[0.0, 1.0]])

    restored = _restore_vqvla_oxe_actions(
        normalized,
        q01=torch.tensor([1.0, 0.0]),
        q99=torch.tensor([5.0, 1.0]),
        normalization_mask=torch.tensor([True, False]),
        eps=1e-8,
    )

    assert torch.allclose(restored, torch.tensor([[3.0, 1.0]]))


def test_vqvae_checkpoint_loader_accepts_numpy_training_metadata(tmp_path):
    checkpoint_path = tmp_path / "action_vqvae_training_checkpoint.pt"
    numpy_rng = np.random.RandomState(0).get_state()
    torch.save(
        {
            "config": {"horizon": 2, "action_dim": 1},
            "model": {"weight": torch.ones(1)},
            "rng_state": {"numpy": numpy_rng},
        },
        checkpoint_path,
    )

    loaded = _load_checkpoint(checkpoint_path)

    assert loaded["config"] == {"horizon": 2, "action_dim": 1}
    assert torch.equal(loaded["model"]["weight"], torch.ones(1))
    assert np.array_equal(loaded["rng_state"]["numpy"][1], numpy_rng[1])


def test_vqvae_loader_decodes_only_the_q0_embedding(tmp_path):
    decoder = VQVLALikeDecoder(
        horizon=2,
        action_dim=1,
        latent_dim=2,
        block_out_channels=(2,),
        layers_per_block=(0,),
        encoder_out_channels=1,
        bottleneck_hw=(2, 1),
        norm_groups=1,
        dropout=0.0,
        num_res_blocks=0,
    )
    q0_codebook = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    checkpoint_path = tmp_path / "action_vqvae.pt"
    torch.save(
        {
            "config": {
                "horizon": 2,
                "action_dim": 1,
                "latent_dim": 2,
                "num_quantizers": 4,
                "codebook_size": 3,
                "block_out_channels": (2,),
                "layers_per_block": (0,),
                "encoder_out_channels": 1,
                "norm_groups": 1,
                "dropout": 0.0,
                "num_res_blocks": 0,
                "normalize_actions": False,
            },
            "model": {
                **{f"decoder.{key}": value for key, value in decoder.state_dict().items()},
                "quantizer.layers.0.codebook": q0_codebook,
                "action_mean": torch.zeros(1),
                "action_std": torch.ones(1),
            },
        },
        checkpoint_path,
    )

    loaded = load_action_vqvae_q0_decoder(checkpoint_path)
    q0_codes = torch.tensor([0, 2])

    assert torch.allclose(
        loaded(q0_codes),
        decoder(q0_codebook[q0_codes]),
    )
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_vqvae_q0_encoder_loader_matches_nearest_codebook_entry(tmp_path):
    encoder = VQVLALikeEncoder(
        horizon=2,
        action_dim=1,
        in_channels=1,
        latent_dim=2,
        block_out_channels=(2,),
        layers_per_block=(0,),
        encoder_out_channels=1,
        norm_groups=1,
        dropout=0.0,
        num_res_blocks=0,
    )
    q0_codebook = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    checkpoint_path = tmp_path / "action_vqvae_encoder.pt"
    torch.save(
        {
            "config": {
                "horizon": 2,
                "action_dim": 1,
                "latent_dim": 2,
                "codebook_size": 3,
                "block_out_channels": (2,),
                "layers_per_block": (0,),
                "encoder_out_channels": 1,
                "norm_groups": 1,
                "dropout": 0.0,
                "num_res_blocks": 0,
                "normalize_actions": False,
            },
            "model": {
                **{f"encoder.{key}": value for key, value in encoder.state_dict().items()},
                "quantizer.layers.0.codebook": q0_codebook,
                "action_mean": torch.zeros(1),
                "action_std": torch.ones(1),
            },
        },
        checkpoint_path,
    )

    loaded = load_action_vqvae_q0_encoder(checkpoint_path)
    actions = torch.tensor([[[-0.5], [0.25]], [[0.5], [-0.25]]])
    with torch.inference_mode():
        latents = encoder(actions.unsqueeze(1))
        expected_distances = torch.cdist(latents, q0_codebook).square()
        expected = expected_distances.argmin(dim=-1)

    assert torch.equal(loaded(actions), expected)
    assert torch.allclose(loaded.compute_code_distances(actions), expected_distances, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


class _DummyPaliGemma:
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        values = tokens.to(torch.float32)
        return torch.stack([values, values], dim=-1)


class _DummyActionMem:
    def __init__(self):
        self.paligemma_with_expert = _DummyPaliGemma()
        self.action_token_map = type("TokenMap", (), {"action_query_token_id": 9})()
        self.state_token_proj = nn.Linear(2, 2, bias=False)
        self.state_token_proj.weight.data.copy_(torch.eye(2))

    def _apply_checkpoint(self, func, *args, **kwargs):
        return func(*args, **kwargs)


def test_embed_prefix_places_state_immediately_before_action_query():
    model = _DummyActionMem()
    images = [torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])]
    image_masks = [torch.tensor([True])]
    language_tokens = torch.tensor([[10, 11, 0]])
    language_masks = torch.tensor([[True, True, False]])
    action_tokens = torch.tensor([[7, 8, 9, 256981]])
    action_masks = torch.tensor([[True, True, True, True]])

    embeddings, pad_masks, attention_blocks = ActionMemPytorch.embed_prefix(
        model,
        images,
        image_masks,
        language_tokens,
        language_masks,
        action_tokens,
        action_masks,
        state=torch.tensor([[3.0, 4.0]]),
    )

    assert embeddings.shape == (1, 10, 2)
    assert torch.equal(embeddings[0, 5:7, 0], torch.tensor([7.0, 8.0]))
    assert torch.equal(embeddings[0, 7], torch.tensor([3.0, 4.0]))
    assert torch.equal(embeddings[0, 8:, 0], torch.tensor([9.0, 256981.0]))
    assert torch.equal(
        pad_masks,
        torch.tensor([[True, True, True, True, False, True, True, True, True, True]]),
    )
    assert torch.equal(
        attention_blocks,
        torch.tensor([[False, False, False, False, False, True, True, True, True, True]]),
    )


def test_masked_action_token_cross_entropy_uses_only_valid_targets():
    logits = torch.tensor(
        [
            [0.0, 0.0, 5.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor([2, 4])
    target_mask = torch.tensor([True, False])

    per_sample_loss, mean_loss, accuracy = _masked_action_token_cross_entropy(
        logits,
        targets,
        target_mask,
    )

    expected = F.cross_entropy(logits[:1], targets[:1])
    assert torch.allclose(per_sample_loss, torch.tensor([expected, 0.0]))
    assert torch.allclose(mean_loss, expected)
    assert accuracy.item() == 1.0


def test_select_action_token_restricts_generation_to_action_vocabulary():
    logits = torch.zeros(2, 10)
    logits[:, 1] = 100.0
    logits[0, 7] = 3.0
    logits[1, 8] = 4.0

    selected = _select_action_token(logits, token_id_min=6, token_id_max=8)

    assert torch.equal(selected, torch.tensor([[7], [8]]))


class _DummyQ0Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("q0_codebook", torch.zeros(3, 1))

    def forward(self, q0_codes):
        return q0_codes.float().view(-1, 1, 1).expand(-1, 2, 1)


def test_training_flow_source_decodes_valid_targets_and_falls_back_for_invalid():
    model = ActionMemPytorch.__new__(ActionMemPytorch)
    nn.Module.__init__(model)
    model.config = type(
        "Config",
        (),
        {
            "max_action_dim": 2,
            "action_vqvae_flow_normalization_eps": 1e-8,
        },
    )()
    model.action_token_map = type(
        "TokenMap",
        (),
        {
            "anchor_token_id": 10,
            "token_id_min": 8,
            "token_id_max": 10,
        },
    )()
    model._normalize_action_vqvae_flow_source = True
    model.register_buffer("_action_vqvae_input_q01", torch.tensor([-1.0]), persistent=False)
    model.register_buffer("_action_vqvae_input_q99", torch.tensor([1.0]), persistent=False)
    model.register_buffer("_action_vqvae_input_mask", torch.tensor([True]), persistent=False)
    model.register_buffer("_action_vqvae_flow_mean", torch.tensor([1.0]), persistent=False)
    model.register_buffer("_action_vqvae_flow_std", torch.tensor([2.0]), persistent=False)
    object.__setattr__(model, "_action_vqvae", _DummyQ0Decoder())
    model.sample_noise = lambda shape, device: torch.full(shape, -1.0, device=device)

    source = model._make_training_flow_source(
        action_tokens=torch.tensor([[7, 9, 10], [7, 9, 8], [7, 9, 0]]),
        action_token_masks=torch.tensor([[True, True, True], [True, True, True], [True, True, False]]),
        actions=torch.zeros(3, 2, 2),
        input_q01=torch.tensor([[-1.0], [0.0], [100.0]]),
        input_q99=torch.tensor([[1.0], [2.0], [200.0]]),
        input_mask=torch.tensor([[True], [True], [True]]),
    )

    assert torch.allclose(source[0], torch.tensor([[-0.5, 0.0], [-0.5, 0.0]]))
    assert torch.allclose(source[1], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.equal(source[2], torch.full((2, 2), -1.0))


def test_vlm_only_core_forward_conditions_on_state_and_skips_flow_source():
    model = ActionMemPytorch.__new__(ActionMemPytorch)
    nn.Module.__init__(model)
    action_tokens = torch.tensor([[7, 9, 10], [7, 9, 8]])
    action_token_masks = torch.ones_like(action_tokens, dtype=torch.bool)
    prefix = (
        torch.zeros(2, 3, 4),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3),
    )
    expected = {"action_token_ce_loss": torch.tensor(1.0)}

    model._validate_action_token_sequence = lambda *_: None
    captured = {}

    def embed_prefix(*args, **kwargs):
        captured["state"] = kwargs["state"]
        return prefix

    model.embed_prefix = embed_prefix
    model._forward_action_token_only = lambda *args: expected
    model._make_training_flow_source = lambda *_: pytest.fail("VLM-only must not build a flow source")

    output = model.forward(
        images=[],
        img_masks=[],
        lang_tokens=torch.zeros(2, 1, dtype=torch.long),
        lang_masks=torch.ones(2, 1, dtype=torch.bool),
        action_tokens=action_tokens,
        action_token_masks=action_token_masks,
        state=torch.ones(2, 4),
        compute_flow=False,
        compute_action_token=True,
    )

    assert output is expected
    assert torch.equal(captured["state"], torch.ones(2, 4))


def test_gradient_checkpointing_configures_paligemma_decoder_layers():
    model = ActionMemPytorch.__new__(ActionMemPytorch)
    nn.Module.__init__(model)
    language_model = PiGemmaModel(
        GemmaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    vision_tower = type("VisionTower", (), {"gradient_checkpointing": False})()
    expert_model = type("ExpertModel", (), {"gradient_checkpointing": False})()
    model.paligemma_with_expert = type(
        "PaliGemmaWithExpert",
        (),
        {
            "paligemma": type(
                "PaliGemma",
                (),
                {
                    "model": type(
                        "PaliGemmaModel",
                        (),
                        {
                            "language_model": language_model,
                            "vision_tower": vision_tower,
                        },
                    )()
                },
            )(),
            "gemma_expert": type("GemmaExpert", (), {"model": expert_model})(),
        },
    )()

    model.gradient_checkpointing_enable()

    assert model.gradient_checkpointing_enabled
    assert language_model.gradient_checkpointing
    assert all(layer.gradient_checkpointing for layer in language_model.layers)
    assert vision_tower.gradient_checkpointing
    assert expert_model.gradient_checkpointing

    model.gradient_checkpointing_disable()

    assert not model.gradient_checkpointing_enabled
    assert not language_model.gradient_checkpointing
    assert not any(layer.gradient_checkpointing for layer in language_model.layers)
    assert not vision_tower.gradient_checkpointing
    assert not expert_model.gradient_checkpointing


class _DummyTrainingCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.states = []

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.zeros(batch_size, device=device)

    def forward(
        self,
        images,
        image_masks,
        language_tokens,
        language_masks,
        action_tokens,
        action_token_masks,
        state,
        actions,
        noise,
        time,
        *,
        action_vqvae_input_q01=None,
        action_vqvae_input_q99=None,
        action_vqvae_input_mask=None,
        compute_flow,
        compute_action_token,
    ):
        self.calls.append((compute_flow, compute_action_token))
        self.states.append(state)
        device = actions.device if actions is not None else action_tokens.device
        del (
            images,
            image_masks,
            language_tokens,
            language_masks,
            action_tokens,
            action_token_masks,
            noise,
            time,
            action_vqvae_input_q01,
            action_vqvae_input_q99,
            action_vqvae_input_mask,
        )
        output = {}
        if compute_flow:
            output["flow_losses"] = torch.full_like(actions, 2.0)
        if compute_action_token:
            output.update(
                {
                    "action_token_loss_per_sample": torch.tensor([3.0, 0.0], device=device),
                    "action_token_target_mask": torch.tensor([True, False], device=device),
                    "action_token_ce_loss": torch.tensor(3.0, device=device),
                    "action_token_accuracy": torch.tensor(0.5, device=device),
                }
            )
        return output


@pytest.mark.parametrize(
    ("training_stage", "expected_scalar", "expected_per_sample", "expected_call"),
    [
        ("vlm_only", 1.5, torch.tensor([3.0, 0.0]), (False, True)),
        ("action_expert_only", 4.0, torch.tensor([4.0, 4.0]), (True, False)),
        ("joint", 5.5, torch.tensor([7.0, 4.0]), (True, True)),
    ],
)
def test_policy_training_stages_select_objectives_for_both_reductions(
    training_stage,
    expected_scalar,
    expected_per_sample,
    expected_call,
):
    policy = ActionMemPolicy.__new__(ActionMemPolicy)
    nn.Module.__init__(policy)
    policy.model = _DummyTrainingCore()
    policy.config = type(
        "Config",
        (),
        {
            "training_stage": training_stage,
            "flow_loss_weight": 2.0,
            "action_token_loss_weight": 0.5,
            "output_features": {ACTION: type("Feature", (), {"shape": (2,)})()},
        },
    )()
    policy._preprocess_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: batch[ACTION]

    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION_TOKENS: torch.tensor([[7, 8, 9, 257023], [7, 8, 9, 0]]),
        ACTION_TOKEN_MASK: torch.tensor([[True, True, True, True], [True, True, True, False]]),
        OBS_STATE: torch.zeros(2, 2),
        ACTION: torch.zeros(2, 1, 2),
    }

    scalar_loss, scalar_metrics = policy.forward(batch)
    per_sample_loss, per_sample_metrics = policy.forward(batch, reduction="none")

    assert scalar_loss.item() == expected_scalar
    assert torch.equal(per_sample_loss, expected_per_sample)
    assert policy.model.calls == [expected_call, expected_call]
    assert all(torch.equal(state, batch[OBS_STATE]) for state in policy.model.states)
    if training_stage != "vlm_only":
        assert scalar_metrics["flow_loss"] == 2.0
    else:
        assert "flow_loss" not in scalar_metrics
    if training_stage != "action_expert_only":
        assert scalar_metrics["action_token_ce_loss"] == 3.0
        assert scalar_metrics["action_token_accuracy"] == 0.5
    else:
        assert "action_token_ce_loss" not in scalar_metrics
    assert per_sample_metrics["loss"] == scalar_metrics["loss"]


def test_actionmem_config_validates_training_stages_and_legacy_alias():
    assert ActionMemConfig(training_stage="vlm_only").training_stage == "vlm_only"
    assert ActionMemConfig(train_expert_only=True).training_stage == "action_expert_only"

    with pytest.raises(ValueError, match="training_stage must be one of"):
        ActionMemConfig(training_stage="unknown")

    with pytest.raises(ValueError, match="conflicts"):
        ActionMemConfig(training_stage="vlm_only", train_expert_only=True)


class _StageFreezeModel(ActionMemPytorch):
    def __init__(self, training_stage):
        nn.Module.__init__(self)
        self.config = type("Config", (), {"training_stage": training_stage})()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
        self.state_token_proj = nn.Linear(2, 2)
        self.state_proj = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.action_time_mlp_in = nn.Linear(2, 2)
        self.action_time_mlp_out = nn.Linear(2, 2)


@pytest.mark.parametrize(
    ("training_stage", "vlm_trainable", "expert_trainable"),
    [
        ("vlm_only", True, False),
        ("action_expert_only", False, True),
        ("joint", True, True),
    ],
)
def test_training_stage_freezes_the_inactive_branch(
    training_stage,
    vlm_trainable,
    expert_trainable,
):
    model = _StageFreezeModel(training_stage)
    model.configure_training_stage()

    vlm_parameters = [
        parameter for name, parameter in model.named_parameters() if model._is_vlm_parameter(name)
    ]
    expert_parameters = [
        parameter for name, parameter in model.named_parameters() if model._is_action_expert_parameter(name)
    ]
    assert vlm_parameters and expert_parameters
    assert all(parameter.requires_grad is vlm_trainable for parameter in vlm_parameters)
    assert all(parameter.requires_grad is expert_trainable for parameter in expert_parameters)


@pytest.mark.parametrize(
    ("training_stage", "state_token_targeted", "expert_state_targeted"),
    [
        ("vlm_only", True, False),
        ("action_expert_only", False, True),
        ("joint", True, True),
    ],
)
def test_default_peft_targets_cover_stage_specific_state_projections(
    training_stage,
    state_token_targeted,
    expert_state_targeted,
):
    policy = ActionMemPolicy.__new__(ActionMemPolicy)
    policy.config = type("Config", (), {"training_stage": training_stage})()
    pattern = policy._get_default_peft_targets()["target_modules"]

    assert bool(re.fullmatch(pattern, "model.state_token_proj")) is state_token_targeted
    assert bool(re.fullmatch(pattern, "model.state_proj")) is expert_state_targeted


class _RestrictedHead(nn.Module):
    def forward(self, hidden_states):
        logits = torch.zeros(hidden_states.shape[0], 10, device=hidden_states.device)
        logits[:, 1] = 100.0
        logits[:, 7] = 3.0
        return logits


class _DummyInferencePaliGemma(nn.Module):
    def __init__(self):
        super().__init__()
        q_proj = type("QProj", (), {"weight": torch.empty(1, dtype=torch.float32)})()
        self.paligemma = type(
            "PaliGemma",
            (),
            {
                "lm_head": _RestrictedHead(),
                "model": type(
                    "PaliGemmaModel",
                    (),
                    {
                        "language_model": type(
                            "LanguageModel",
                            (),
                            {
                                "config": type("Config", (), {"_attn_implementation": "eager"})(),
                                "layers": [
                                    type(
                                        "Layer",
                                        (),
                                        {
                                            "self_attn": type(
                                                "Attention",
                                                (),
                                                {"q_proj": q_proj},
                                            )()
                                        },
                                    )()
                                ],
                            },
                        )()
                    },
                )(),
            },
        )()
        self.forward_inputs = []

    def embed_language_tokens(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2)

    def forward(self, *, inputs_embeds, past_key_values, **kwargs):
        del kwargs
        self.forward_inputs.append(inputs_embeds[0].detach().clone())
        cache = "prefill" if past_key_values is None else "with_generated_token"
        return [inputs_embeds[0], None], cache


def test_vlm_only_uses_layer_checkpointing_without_a_whole_model_checkpoint():
    model = ActionMemPytorch.__new__(ActionMemPytorch)
    nn.Module.__init__(model)
    model.paligemma_with_expert = _DummyInferencePaliGemma()
    model._apply_checkpoint = lambda *_args, **_kwargs: pytest.fail(
        "VLM-only must not checkpoint the whole PaliGemma forward"
    )

    output = model._forward_action_token_only(
        prefix_embs=torch.ones(2, 4, 2),
        prefix_pad_masks=torch.ones(2, 4, dtype=torch.bool),
        prefix_att_masks=torch.zeros(2, 4, dtype=torch.bool),
        action_tokens=torch.tensor([[1, 2, 3, 7], [1, 2, 3, 7]]),
        action_token_masks=torch.ones(2, 4, dtype=torch.bool),
    )

    assert len(model.paligemma_with_expert.forward_inputs) == 1
    assert output["action_token_ce_loss"].ndim == 0


def test_inference_generates_restricted_token_and_adds_it_to_prefix_cache(monkeypatch):
    model = ActionMemPytorch.__new__(ActionMemPytorch)
    nn.Module.__init__(model)
    model.config = type(
        "Config",
        (),
        {
            "num_inference_steps": 1,
            "chunk_size": 2,
            "max_action_dim": 2,
            "rtc_config": None,
        },
    )()
    model.action_token_map = type(
        "TokenMap",
        (),
        {"token_id_min": 6, "token_id_max": 8, "action_query_token_id": 9},
    )()
    model.rtc_processor = None
    model.paligemma_with_expert = _DummyInferencePaliGemma()
    model.embed_prefix = lambda *args, **kwargs: (
        torch.ones(1, 5, 2),
        torch.ones(1, 5, dtype=torch.bool),
        torch.zeros(1, 5, dtype=torch.bool),
    )

    monkeypatch.setattr(
        modeling_actionmem,
        "euler_integrate",
        lambda denoise_fn, noise, num_steps, **kwargs: noise,
    )
    decoded_initial_actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model.decode_action_tokens = lambda tokens: decoded_initial_actions

    output = model.sample_actions(
        images=[],
        img_masks=[],
        lang_tokens=torch.ones(1, 3, dtype=torch.long),
        lang_masks=torch.ones(1, 3, dtype=torch.bool),
        action_tokens=torch.tensor([[7, 8, 9, 0]]),
        action_token_masks=torch.tensor([[True, True, True, False]]),
        state=torch.zeros(1, 2),
    )

    assert torch.equal(output, decoded_initial_actions)
    assert len(model.paligemma_with_expert.forward_inputs) == 2
    generated_embedding = model.paligemma_with_expert.forward_inputs[1]
    assert torch.equal(generated_embedding, torch.full((1, 1, 2), 7.0))
