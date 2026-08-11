import re

import pytest
import torch
from torch import nn

import lerobot.policies.smol_actionmem.modeling_smol_actionmem as modeling_smol_actionmem
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.smol_actionmem.configuration_smol_actionmem import SmolActionMemConfig
from lerobot.policies.smol_actionmem.modeling_smol_actionmem import (
    SmolActionMemFlowMatching,
    SmolActionMemPolicy,
)
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class _DummyVLM:
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        return tokens.float().unsqueeze(-1).expand(-1, -1, 4)


def test_embed_prefix_orders_state_before_action_tokens():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    model.vlm_with_expert = _DummyVLM()
    model.action_token_map = type("Map", (), {"action_query_token_id": 358})()
    model.add_image_special_tokens = False
    model.prefix_length = -1
    model.state_proj = nn.Linear(2, 4, bias=False)
    model.state_proj.weight.data.copy_(torch.eye(4, 2))

    embeddings, padding, blocks, query_positions = model.embed_prefix(
        images=[torch.ones(1, 2, 4)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[10, 11, 2]]),
        lang_masks=torch.tensor([[True, True, False]]),
        action_tokens=torch.tensor([[356, 357, 358, 355]]),
        action_token_masks=torch.ones(1, 4, dtype=torch.bool),
        state=torch.tensor([[1.0, 2.0]]),
    )

    assert embeddings.shape == (1, 10, 4)
    assert torch.equal(embeddings[0, 5:7, 0] / 2, torch.tensor([356.0, 357.0]))
    assert torch.equal(embeddings[0, 7], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert torch.equal(embeddings[0, 8:10, 0] / 2, torch.tensor([358.0, 355.0]))
    assert query_positions.item() == 8
    assert torch.equal(
        padding,
        torch.tensor([[True, True, True, True, False, True, True, True, True, True]]),
    )
    assert torch.equal(
        blocks,
        torch.tensor([[False, False, False, False, False, True, True, True, True, True]]),
    )


def test_training_flow_source_is_standard_gaussian_noise():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    actions = torch.zeros(2, 16, 7)
    sampled_noise = torch.randn_like(actions)
    calls = []

    def sample_noise(shape, device):
        calls.append((shape, device))
        return sampled_noise

    model.sample_noise = sample_noise

    source = model._make_training_flow_source(actions)

    assert source is sampled_noise
    assert calls == [(actions.shape, actions.device)]


class _StageModel(SmolActionMemFlowMatching):
    def __init__(self, stage):
        nn.Module.__init__(self)
        self.config = type("Config", (), {"training_stage": stage, "freeze_vision_encoder": False})()
        self.vlm_with_expert = nn.Module()
        self.vlm_with_expert.vlm = nn.Linear(2, 2)
        self.vlm_with_expert.lm_expert = nn.Linear(2, 2)
        self.state_proj = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.action_time_mlp_in = nn.Linear(2, 2)
        self.action_time_mlp_out = nn.Linear(2, 2)
        self._training_stage_configured = False


@pytest.mark.parametrize(
    ("stage", "vlm_trainable", "expert_trainable"),
    [
        ("vlm_only", True, False),
        ("action_expert_only", False, True),
        ("joint", True, True),
    ],
)
def test_training_stage_selects_branch(stage, vlm_trainable, expert_trainable):
    model = _StageModel(stage)
    model.configure_training_stage()
    vlm = [
        parameter for name, parameter in model.named_parameters() if name.startswith("vlm_with_expert.vlm.")
    ]
    expert = [
        parameter
        for name, parameter in model.named_parameters()
        if model._is_action_expert_parameter(name) and not name.startswith("state_proj.")
    ]
    state_projection = [
        parameter for name, parameter in model.named_parameters() if name.startswith("state_proj.")
    ]
    assert vlm and expert
    assert all(parameter.requires_grad is vlm_trainable for parameter in vlm)
    assert all(parameter.requires_grad is expert_trainable for parameter in expert)
    assert state_projection and all(parameter.requires_grad for parameter in state_projection)


class _DummyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.states = []

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
        compute_flow,
        compute_action_token,
        action_vqvae_input_q01=None,
        action_vqvae_input_q99=None,
        action_vqvae_input_mask=None,
    ):
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
        self.calls.append((compute_flow, compute_action_token))
        self.states.append(state)
        device = actions.device if actions is not None else torch.device("cpu")
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
    ("stage", "expected", "call"),
    [
        ("vlm_only", 1.5, (False, True)),
        ("action_expert_only", 4.0, (True, False)),
        ("joint", 5.5, (True, True)),
    ],
)
def test_policy_training_stages_select_losses(stage, expected, call):
    policy = SmolActionMemPolicy.__new__(SmolActionMemPolicy)
    nn.Module.__init__(policy)
    policy.model = _DummyCore()
    policy.config = type(
        "Config",
        (),
        {
            "training_stage": stage,
            "flow_loss_weight": 2.0,
            "action_token_loss_weight": 0.5,
            "adapt_to_pi_aloha": False,
            "action_feature": type("Feature", (), {"shape": (2,)})(),
        },
    )()
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: batch[ACTION]
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION_TOKENS: torch.tensor([[356, 357, 358, 355], [356, 357, 358, 2]]),
        ACTION_TOKEN_MASK: torch.tensor([[True, True, True, True], [True, True, True, False]]),
        OBS_STATE: torch.zeros(2, 2),
        ACTION: torch.zeros(2, 1, 2),
    }

    loss, metrics = policy.forward(batch)
    assert loss.item() == expected
    assert policy.model.calls == [call]
    assert torch.equal(policy.model.states[0], batch[OBS_STATE])
    assert metrics["loss"] == expected


def test_config_validates_training_stage_and_loss_weights():
    assert SmolActionMemConfig(training_stage="vlm_only").chunk_size == 16
    assert SmolActionMemConfig(train_expert_only=True).training_stage == "action_expert_only"
    with pytest.raises(ValueError, match="training_stage must be one of"):
        SmolActionMemConfig(training_stage="unknown")
    with pytest.raises(ValueError, match="requires action_token_loss_weight"):
        SmolActionMemConfig(training_stage="vlm_only", action_token_loss_weight=0)


def test_policy_is_registered_by_naming_convention():
    config = make_policy_config("smol_actionmem")
    assert isinstance(config, SmolActionMemConfig)
    assert get_policy_class("smol_actionmem") is SmolActionMemPolicy


def test_default_peft_targets_cover_both_branches_and_projections():
    policy = SmolActionMemPolicy.__new__(SmolActionMemPolicy)
    pattern = policy._get_default_peft_targets()["target_modules"]
    expected_modules = [
        "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj",
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens",
        "model.vlm_with_expert.vlm.lm_head",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.v_proj",
        "model.state_proj",
        "model.action_out_proj",
    ]
    assert all(re.fullmatch(pattern, module) for module in expected_modules)
    assert not re.fullmatch(pattern, "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0")


class _InferenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(10, 2))

    def forward(self, hidden):
        logits = torch.zeros(*hidden.shape[:-1], 10, device=hidden.device)
        logits[..., 7] = 10
        return logits


class _InferenceVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_head = _InferenceHead()
        self.vlm = type("VLM", (), {"get_output_embeddings": lambda _: self.output_head})()
        self.forward_calls = 0

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        self.forward_calls += 1
        return (inputs_embeds[0], None), "prefix_cache"


def test_inference_uses_generated_code_condition_with_gaussian_noise(monkeypatch):
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    model.config = type(
        "Config",
        (),
        {
            "num_inference_steps": 2,
            "chunk_size": 2,
            "max_action_dim": 2,
            "use_cache": True,
            "rtc_config": None,
        },
    )()
    model.action_token_map = type(
        "TokenMap",
        (),
        {"token_id_min": 6, "token_id_max": 8, "action_query_token_id": 9},
    )()
    model.rtc_processor = None
    model.vlm_with_expert = _InferenceVLM()
    embedded_action_sequences = []

    def embed_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_tokens,
        action_token_masks,
        state,
    ):
        del images, img_masks, lang_tokens, lang_masks, action_token_masks, state
        embedded_action_sequences.append(action_tokens.clone())
        return (
            torch.ones(1, 5, 2),
            torch.ones(1, 5, dtype=torch.bool),
            torch.zeros(1, 5, dtype=torch.bool),
            torch.tensor([4]),
        )

    model.embed_prefix = embed_prefix
    sampled_noise = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model.sample_noise = lambda shape, device: sampled_noise
    monkeypatch.setattr(
        modeling_smol_actionmem,
        "euler_integrate",
        lambda denoise_fn, noise, num_steps, **kwargs: noise,
    )

    output = model.sample_actions(
        images=[],
        img_masks=[],
        lang_tokens=torch.ones(1, 3, dtype=torch.long),
        lang_masks=torch.ones(1, 3, dtype=torch.bool),
        action_tokens=torch.tensor([[8, 9, 0]]),
        action_token_masks=torch.tensor([[True, True, False]]),
        state=torch.zeros(1, 2),
    )

    assert torch.equal(output, sampled_noise)
    assert model.vlm_with_expert.forward_calls == 2
    assert torch.equal(embedded_action_sequences[0], torch.tensor([[8, 9]]))
    assert torch.equal(embedded_action_sequences[1], torch.tensor([[8, 9, 7]]))
