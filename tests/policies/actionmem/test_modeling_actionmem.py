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

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

import lerobot.policies.actionmem.modeling_actionmem as modeling_actionmem
from lerobot.configs import NormalizationMode
from lerobot.policies.actionmem.action_vqvae import (
    VQVLALikeDecoder,
    load_action_vqvae_q0_decoder,
)
from lerobot.policies.actionmem.modeling_actionmem import (
    ActionMemPolicy,
    ActionMemPytorch,
    _configure_action_vqvae_flow_normalization,
    _masked_action_token_cross_entropy,
    _select_action_token,
)
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
            "action_vqvae_flow_mean": [100.0, 100.0],
            "action_vqvae_flow_std": [100.0, 100.0],
        },
    )()
    dataset_stats = {
        ACTION: {
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([3.0, 4.0]),
        }
    }

    _configure_action_vqvae_flow_normalization(config, dataset_stats)

    assert config.action_vqvae_flow_mean == [1.0, 2.0]
    assert config.action_vqvae_flow_std == [3.0, 4.0]


def test_flow_source_normalization_reuses_saved_stats_without_dataset_metadata():
    config = type(
        "Config",
        (),
        {
            "normalization_mapping": {"ACTION": NormalizationMode.MEAN_STD},
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
            "action_vqvae_flow_mean": None,
            "action_vqvae_flow_std": None,
        },
    )()

    with pytest.raises(ValueError, match="mean/std are unavailable"):
        _configure_action_vqvae_flow_normalization(config, dataset_stats=None)


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


class _DummyPaliGemma:
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        values = tokens.to(torch.float32)
        return torch.stack([values, values], dim=-1)


class _DummyActionMem:
    def __init__(self):
        self.paligemma_with_expert = _DummyPaliGemma()

    def _apply_checkpoint(self, func, *args, **kwargs):
        return func(*args, **kwargs)


def test_embed_prefix_places_causal_action_tokens_after_task_tokens():
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
    )

    assert embeddings.shape == (1, 9, 2)
    assert torch.equal(embeddings[0, -4:, 0], action_tokens[0].float())
    assert torch.equal(
        pad_masks,
        torch.tensor([[True, True, True, True, False, True, True, True, True]]),
    )
    assert torch.equal(
        attention_blocks,
        torch.tensor([[False, False, False, False, False, True, True, True, True]]),
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
    model.register_buffer("_action_vqvae_flow_mean", torch.tensor([1.0]), persistent=False)
    model.register_buffer("_action_vqvae_flow_std", torch.tensor([2.0]), persistent=False)
    object.__setattr__(model, "_action_vqvae", _DummyQ0Decoder())
    model.sample_noise = lambda shape, device: torch.full(shape, -1.0, device=device)

    source = model._make_training_flow_source(
        action_tokens=torch.tensor([[7, 9, 10], [7, 9, 8], [7, 9, 0]]),
        action_token_masks=torch.tensor([[True, True, True], [True, True, True], [True, True, False]]),
        actions=torch.zeros(3, 2, 2),
    )

    assert torch.allclose(source[0], torch.tensor([[-0.5, 0.0], [-0.5, 0.0]]))
    assert torch.allclose(source[1], torch.tensor([[0.5, 0.0], [0.5, 0.0]]))
    assert torch.equal(source[2], torch.full((2, 2), -1.0))


class _DummyTrainingCore(nn.Module):
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
    ):
        del (
            images,
            image_masks,
            language_tokens,
            language_masks,
            action_tokens,
            action_token_masks,
            state,
            noise,
            time,
        )
        return {
            "flow_losses": torch.full_like(actions, 2.0),
            "action_token_loss_per_sample": torch.tensor([3.0, 0.0], device=actions.device),
            "action_token_target_mask": torch.tensor([True, False], device=actions.device),
            "action_token_ce_loss": torch.tensor(3.0, device=actions.device),
            "action_token_accuracy": torch.tensor(0.5, device=actions.device),
        }


def test_policy_combines_flow_and_action_token_losses_for_both_reductions():
    policy = ActionMemPolicy.__new__(ActionMemPolicy)
    nn.Module.__init__(policy)
    policy.model = _DummyTrainingCore()
    policy.config = type(
        "Config",
        (),
        {
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

    assert scalar_loss.item() == 3.5
    assert torch.equal(per_sample_loss, torch.tensor([5.0, 2.0]))
    assert scalar_metrics["flow_loss"] == 2.0
    assert scalar_metrics["action_token_ce_loss"] == 3.0
    assert scalar_metrics["action_token_accuracy"] == 0.5
    assert per_sample_metrics["loss"] == scalar_metrics["loss"]


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
    model.embed_prefix = lambda *args: (
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
