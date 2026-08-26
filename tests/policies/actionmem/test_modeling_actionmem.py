import re

import pytest
import torch
from torch import nn
from transformers.models.gemma.configuration_gemma import GemmaConfig

from lerobot.policies.action_code import ActionCodeLayout
from lerobot.policies.actionmem.configuration_actionmem import ActionMemConfig
from lerobot.policies.actionmem.modeling_actionmem import (
    ActionMemPolicy,
    ActionMemPytorch,
    PaliGemmaWithExpertModel,
)
from lerobot.policies.pi_gemma import PiGemmaModel
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_DISTANCES,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class _DummyPaliGemma:
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        values = tokens.to(torch.float32)
        return torch.stack([values, values], dim=-1)


def test_actionmem_fsdp_keeps_uniform_fp32_master_parameters(monkeypatch):
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.projection = nn.Linear(2, 2)

    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    model.to_bfloat16_for_selected_params("bfloat16")

    assert model.projection.weight.dtype == torch.float32


def test_actionmem_non_fsdp_uses_configured_bfloat16(monkeypatch):
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.projection = nn.Linear(2, 2)

    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    model.to_bfloat16_for_selected_params("bfloat16")

    assert model.projection.weight.dtype == torch.bfloat16


class _CodeEmbedding(nn.Module):
    def forward(self, tokens):
        values = tokens.to(torch.float32)
        return torch.stack([values, values], dim=-1)


class _DummyActionMem:
    def __init__(self):
        self.paligemma_with_expert = _DummyPaliGemma()
        self.action_code_layout = ActionCodeLayout()
        self.action_code_embedding = _CodeEmbedding()
        self.state_token_proj = nn.Linear(2, 2, bias=False)
        self.state_token_proj.weight.data.copy_(torch.eye(2))

    def _apply_checkpoint(self, function, *args, **kwargs):
        return function(*args, **kwargs)


def test_embed_prefix_places_state_immediately_before_local_action_query():
    model = _DummyActionMem()
    embeddings, pad_masks, attention_blocks = ActionMemPytorch.embed_prefix(
        model,
        [torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])],
        [torch.tensor([True])],
        torch.tensor([[10, 11, 0]]),
        torch.tensor([[True, True, False]]),
        torch.tensor([[256, 257, 258]]),
        torch.tensor([[True, True, True]]),
        state=torch.tensor([[3.0, 4.0]]),
    )

    assert embeddings.shape == (1, 9, 2)
    assert torch.equal(embeddings[0, 5:7, 0], torch.tensor([256.0, 257.0]))
    assert torch.equal(embeddings[0, 7], torch.tensor([3.0, 4.0]))
    assert embeddings[0, -1, 0].item() == 258.0
    assert torch.equal(
        pad_masks,
        torch.tensor([[True, True, True, True, False, True, True, True, True]]),
    )
    assert torch.equal(
        attention_blocks,
        torch.tensor([[False, False, False, False, False, True, True, True, True]]),
    )


def test_vlm_only_core_forward_conditions_on_state_and_passes_prototype_distances():
    model = ActionMemPytorch.__new__(ActionMemPytorch)
    nn.Module.__init__(model)
    tokens = torch.tensor([[256, 257, 258, 10], [256, 257, 258, 20]])
    masks = torch.ones_like(tokens, dtype=torch.bool)
    distances = torch.rand(2, 256)
    expected = {"action_token_kl_loss": torch.tensor(1.0)}
    captured = {}

    model._validate_action_token_sequence = lambda *_: None

    def embed_prefix(*args, **kwargs):
        captured["state"] = kwargs["state"]
        return (
            torch.zeros(2, 3, 4),
            torch.ones(2, 3, dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.bool),
        )

    def token_only(*args):
        captured["distances"] = args[-1]
        return expected

    model.embed_prefix = embed_prefix
    model._forward_action_token_only = token_only
    output = model.forward(
        images=[],
        img_masks=[],
        lang_tokens=torch.zeros(2, 1, dtype=torch.long),
        lang_masks=torch.ones(2, 1, dtype=torch.bool),
        action_tokens=tokens,
        action_token_masks=masks,
        state=torch.ones(2, 4),
        action_token_distances=distances,
        compute_flow=False,
        compute_action_token=True,
    )

    assert output is expected
    assert torch.equal(captured["state"], torch.ones(2, 4))
    assert captured["distances"] is distances


def test_action_classifier_reads_final_action_query_hidden_state():
    model = ActionMemPytorch.__new__(ActionMemPytorch)
    nn.Module.__init__(model)
    model.action_classifier = nn.Linear(4, 256)
    with torch.no_grad():
        model.action_classifier.weight.zero_()
        model.action_classifier.bias.zero_()
        model.action_classifier.weight[7, 2] = 5.0
        model.action_classifier.weight[8, 2] = -5.0

    logits = model._compute_action_logits(
        prefix_out=torch.tensor(
            [
                [[9.0, 9.0, -9.0, 9.0], [0.0, 0.0, 1.0, 0.0]],
                [[9.0, 9.0, -9.0, 9.0], [0.0, 0.0, 1.0, 0.0]],
            ]
        )
    )

    assert logits.shape == (2, 256)
    assert torch.equal(logits.argmax(dim=-1), torch.tensor([7, 7]))


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
                        {"language_model": language_model, "vision_tower": vision_tower},
                    )()
                },
            )(),
            "gemma_expert": type("GemmaExpert", (), {"model": expert_model})(),
        },
    )()

    model.gradient_checkpointing_enable()
    assert language_model.gradient_checkpointing
    assert all(layer.gradient_checkpointing for layer in language_model.layers)
    assert vision_tower.gradient_checkpointing
    assert expert_model.gradient_checkpointing

    model.gradient_checkpointing_disable()
    assert not language_model.gradient_checkpointing
    assert not any(layer.gradient_checkpointing for layer in language_model.layers)


class _DummyTrainingCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.distances = []

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
        action_token_distances,
        compute_flow,
        compute_action_token,
    ):
        del images, image_masks, language_tokens, language_masks, action_tokens, action_token_masks
        del state, noise, time
        self.calls.append((compute_flow, compute_action_token))
        self.distances.append(action_token_distances)
        device = actions.device if actions is not None else action_token_distances.device
        output = {}
        if compute_flow:
            output.update(
                {
                    "flow_losses": torch.full_like(actions, 2.0),
                    "action_condition_gamma_rms": torch.tensor(0.0, device=device),
                    "action_condition_beta_rms": torch.tensor(0.0, device=device),
                    "action_condition_logit_std": torch.tensor(1.0, device=device),
                    "action_condition_predicted_entropy": torch.tensor(2.0, device=device),
                }
            )
        if compute_action_token:
            output.update(
                {
                    "action_token_loss_per_sample": torch.tensor([3.0, 0.0], device=device),
                    "action_token_target_mask": torch.tensor([True, False], device=device),
                    "action_token_kl_loss": torch.tensor(3.0, device=device),
                    "action_token_accuracy": torch.tensor(0.5, device=device),
                    "action_token_target_rank": torch.tensor(4.0, device=device),
                    "action_token_soft_target_entropy": torch.tensor(1.5, device=device),
                    "action_token_soft_target_peak_probability": torch.tensor(0.4, device=device),
                }
            )
        return output


@pytest.mark.parametrize(
    ("stage", "expected_scalar", "expected_per_sample", "expected_call"),
    [
        ("vlm_only", 1.5, torch.tensor([3.0, 0.0]), (False, True)),
        ("action_expert_only", 4.0, torch.tensor([4.0, 4.0]), (True, False)),
        ("joint", 5.5, torch.tensor([7.0, 4.0]), (True, True)),
    ],
)
def test_policy_training_stages_select_effect_code_and_flow_objectives(
    stage, expected_scalar, expected_per_sample, expected_call
):
    policy = ActionMemPolicy.__new__(ActionMemPolicy)
    nn.Module.__init__(policy)
    policy.model = _DummyTrainingCore()
    policy.config = type(
        "Config",
        (),
        {
            "training_stage": stage,
            "flow_loss_weight": 2.0,
            "action_token_loss_weight": 0.5,
            "output_features": {ACTION: type("Feature", (), {"shape": (2,)})()},
        },
    )()
    policy._preprocess_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: batch[ACTION]
    module_forward_calls = []
    policy.model.register_forward_pre_hook(lambda *_: module_forward_calls.append(True))
    distances = torch.rand(2, 256)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION_TOKENS: torch.tensor([[256, 257, 258, 10], [256, 257, 258, 259]]),
        ACTION_TOKEN_MASK: torch.tensor([[True, True, True, True], [True, True, True, False]]),
        ACTION_TOKEN_DISTANCES: distances,
        OBS_STATE: torch.zeros(2, 2),
        ACTION: torch.zeros(2, 1, 2),
    }

    scalar_loss, metrics = policy.forward(batch)
    per_sample_loss, _ = policy.forward(batch, reduction="none")

    assert scalar_loss.item() == expected_scalar
    assert torch.equal(per_sample_loss, expected_per_sample)
    assert policy.model.calls == [expected_call, expected_call]
    assert len(module_forward_calls) == 2
    assert all(value is distances for value in policy.model.distances)
    if stage != "action_expert_only":
        assert metrics["action_token_kl_loss"] == 3.0


def test_actionmem_config_and_stage_specific_trainability():
    assert ActionMemConfig().chunk_size == 10
    assert ActionMemConfig(training_stage="vlm_only").training_stage == "vlm_only"
    assert ActionMemConfig(train_expert_only=True).training_stage == "action_expert_only"
    with pytest.raises(ValueError, match="training_stage must be one of"):
        ActionMemConfig(training_stage="unknown")

    class StageModel(ActionMemPytorch):
        def __init__(self, stage):
            nn.Module.__init__(self)
            self.config = type("Config", (), {"training_stage": stage})()
            self.paligemma_with_expert = nn.Module()
            self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
            self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
            self.state_token_proj = nn.Linear(2, 2)
            self.state_proj = nn.Linear(2, 2)
            self.action_code_embedding = nn.Embedding(8, 2)
            self.action_classifier = nn.Linear(2, 4)
            self.action_condition_proj = nn.Linear(4, 4)
            self.action_in_proj = nn.Linear(2, 2)
            self.action_out_proj = nn.Linear(2, 2)
            self.action_time_mlp_in = nn.Linear(2, 2)
            self.action_time_mlp_out = nn.Linear(2, 2)

    vlm = StageModel("vlm_only")
    vlm.configure_training_stage()
    assert vlm.paligemma_with_expert.paligemma.weight.requires_grad
    assert not vlm.paligemma_with_expert.gemma_expert.weight.requires_grad
    assert vlm.action_classifier.weight.requires_grad
    assert not vlm.action_condition_proj.weight.requires_grad

    expert = StageModel("action_expert_only")
    expert.configure_training_stage()
    assert not expert.action_classifier.weight.requires_grad
    assert expert.action_condition_proj.weight.requires_grad

    joint = StageModel("joint")
    joint.configure_training_stage()
    assert all(parameter.requires_grad for parameter in joint.parameters())


@pytest.mark.parametrize("stage", ["vlm_only", "action_expert_only", "joint"])
def test_default_peft_targets_save_independent_action_modules(stage):
    policy = ActionMemPolicy.__new__(ActionMemPolicy)
    policy.config = type("Config", (), {"training_stage": stage})()
    targets = policy._get_default_peft_targets()

    assert set(targets["modules_to_save"]) == {
        "action_code_embedding",
        "action_classifier",
        "action_condition_proj",
    }
    assert bool(re.fullmatch(targets["target_modules"], "model.state_token_proj")) is (
        stage != "action_expert_only"
    )
