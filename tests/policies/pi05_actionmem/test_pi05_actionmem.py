import json
import re
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lerobot.policies.pi05_actionmem.modeling_pi05_actionmem as modeling_pi05_actionmem
from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.datasets.rlds_dataset import resolve_actionmem_token_metadata
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pi05_actionmem.configuration_pi05_actionmem import PI05ActionMemConfig
from lerobot.policies.pi05_actionmem.modeling_pi05_actionmem import (
    PI05ActionMemPolicy,
    PI05ActionMemPytorch,
    _configure_action_vqvae_flow_normalization,
)
from lerobot.policies.pi05_actionmem.processor_pi05_actionmem import (
    PI05ActionMemActionTokenProcessorStep,
    PI05ActionMemPrepareStateTokenizerProcessorStep,
    reconcile_pi05_actionmem_processors,
)
from lerobot.processor import DataProcessorPipeline, TokenizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def _write_token_map(tmp_path):
    path = tmp_path / "token_map.json"
    path.write_text(
        json.dumps(
            {
                "vqvae": {
                    "codebook_size": 256,
                    "code_id_min": 0,
                    "code_id_max": 255,
                    "invalid_value": -1,
                    "action_horizon": 16,
                    "action_dim": 7,
                },
                "action_tokens": {
                    "anchor_token_id": 257023,
                    "token_id_min": 256768,
                    "token_id_max": 257023,
                },
                "control_tokens": {
                    "action_memory_start": {"token_id": 7},
                    "action_memory_end": {"token_id": 8},
                    "action_query": {"token_id": 9},
                },
                "padding": {"token_id": 0},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_policy_is_registered_and_keeps_pi05_defaults():
    config = PI05ActionMemConfig()

    assert config.type == "pi05_actionmem"
    assert config.chunk_size == 16
    assert config.tokenizer_max_length == 200
    assert config.normalization_mapping["STATE"] is NormalizationMode.QUANTILES
    assert config.normalization_mapping["ACTION"] is NormalizationMode.QUANTILES
    assert PreTrainedConfig.get_choice_class("pi05_actionmem") is PI05ActionMemConfig
    assert get_policy_class("pi05_actionmem") is PI05ActionMemPolicy


def test_config_saves_token_map_next_to_checkpoint(tmp_path):
    token_map = _write_token_map(tmp_path)
    output_dir = tmp_path / "checkpoint"
    output_dir.mkdir()
    config = PI05ActionMemConfig(action_token_map_path=str(token_map))

    config._save_pretrained(output_dir)

    assert (output_dir / "token_map.json").read_text() == token_map.read_text()
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert saved_config["action_token_map_path"] == "token_map.json"
    assert config.action_token_map_path == str(token_map)


def test_rlds_resolves_pi05_actionmem_token_map_from_model_directory(tmp_path):
    token_map = _write_token_map(tmp_path)
    config = SimpleNamespace(
        type="pi05_actionmem",
        pretrained_path=str(tmp_path),
        action_token_map_path=None,
    )

    metadata = resolve_actionmem_token_metadata(config)

    assert metadata.path == str(token_map.resolve())
    assert metadata.action_horizon == 16
    assert config.action_token_map_path == str(token_map.resolve())


def test_action_token_processor_matches_actionmem_protocol(tmp_path):
    step = PI05ActionMemActionTokenProcessorStep(token_map_path=str(_write_token_map(tmp_path)))
    transition = create_transition(
        observation={OBS_LANGUAGE_TOKENS: torch.ones(2, 4, dtype=torch.long)},
        complementary_data={"task": ["pick", "place"], ACTION_TOKEN: torch.tensor([[0], [255]])},
    )

    output = step(transition)[TransitionKey.COMPLEMENTARY_DATA]

    assert torch.equal(
        output[ACTION_TOKENS],
        torch.tensor([[7, 8, 9, 257023], [7, 8, 9, 256768]]),
    )
    assert output[ACTION_TOKEN_MASK].all()


def test_pi05_state_stays_before_action_memory_query():
    step = PI05ActionMemPrepareStateTokenizerProcessorStep(max_state_dim=3)
    transition = create_transition(
        observation={OBS_STATE: torch.tensor([[-1.0, 0.0, 1.0]])},
        complementary_data={"task": ["pick_up\nthe_cube"]},
    )

    output = step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"]

    assert output == ["Task: pick up the cube, State: 0 128 255;\nAction: "]


def test_copied_pi05_processor_is_upgraded_with_action_tokens(tmp_path):
    tokenizer_step = TokenizerProcessorStep.__new__(TokenizerProcessorStep)
    preprocessor = DataProcessorPipeline(steps=(tokenizer_step,))
    postprocessor = DataProcessorPipeline(steps=())
    config = PI05ActionMemConfig(action_token_map_path=str(_write_token_map(tmp_path)))

    reconciled, _ = reconcile_pi05_actionmem_processors(config, preprocessor, postprocessor)

    assert isinstance(reconciled.steps, list)
    assert reconciled.steps[0] is tokenizer_step
    assert isinstance(reconciled.steps[1], PI05ActionMemActionTokenProcessorStep)


def test_quantile_flow_source_stats_use_runtime_dataset_stats():
    config = type(
        "Config",
        (),
        {
            "normalization_mapping": {"ACTION": NormalizationMode.QUANTILES},
            "action_vqvae_flow_mean": None,
            "action_vqvae_flow_std": None,
            "action_vqvae_flow_q01": [100.0, 100.0],
            "action_vqvae_flow_q99": [200.0, 200.0],
        },
    )()
    dataset_stats = {
        ACTION: {
            "q01": torch.tensor([1.0, 2.0]),
            "q99": torch.tensor([3.0, 6.0]),
        }
    }

    _configure_action_vqvae_flow_normalization(config, dataset_stats)

    assert config.action_vqvae_flow_q01 == [1.0, 2.0]
    assert config.action_vqvae_flow_q99 == [3.0, 6.0]
    assert config.action_vqvae_flow_mean is None
    assert config.action_vqvae_flow_std is None


def test_vqvae_flow_source_uses_pi05_quantile_normalization():
    model = PI05ActionMemPytorch.__new__(PI05ActionMemPytorch)
    nn.Module.__init__(model)
    model.config = type(
        "Config",
        (),
        {"max_action_dim": 3, "action_vqvae_flow_normalization_eps": 1e-8},
    )()
    model.action_token_map = type(
        "TokenMap",
        (),
        {"token_id_min": 90, "token_id_max": 100, "anchor_token_id": 100},
    )()
    model._action_normalization = NormalizationMode.QUANTILES
    model.register_buffer("_action_vqvae_flow_stat_a", torch.tensor([0.0, 10.0]))
    model.register_buffer("_action_vqvae_flow_stat_b", torch.tensor([10.0, 20.0]))
    decoded = torch.tensor([[[0.0, 15.0], [10.0, 20.0]]])
    model._get_action_vqvae = lambda device: lambda q0: decoded.to(device)

    result = model.decode_action_tokens(torch.tensor([100]))

    assert torch.equal(
        result,
        torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]),
    )


class _StageFreezeModel(PI05ActionMemPytorch):
    def __init__(self, stage):
        nn.Module.__init__(self)
        self.config = type("Config", (), {"training_stage": stage})()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.time_mlp_in = nn.Linear(2, 2)
        self.time_mlp_out = nn.Linear(2, 2)


@pytest.mark.parametrize(
    ("stage", "vlm_trainable", "expert_trainable"),
    [
        ("vlm_only", True, False),
        ("action_expert_only", False, True),
        ("joint", True, True),
    ],
)
def test_training_stage_freezes_pi05_branches(stage, vlm_trainable, expert_trainable):
    model = _StageFreezeModel(stage)
    model.configure_training_stage()

    vlm = [parameter for name, parameter in model.named_parameters() if model._is_vlm_parameter(name)]
    expert = [
        parameter for name, parameter in model.named_parameters() if model._is_action_expert_parameter(name)
    ]
    assert vlm and expert
    assert all(parameter.requires_grad is vlm_trainable for parameter in vlm)
    assert all(parameter.requires_grad is expert_trainable for parameter in expert)


class _DummyTrainingCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

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
        actions,
        noise,
        time,
        *,
        compute_flow,
        compute_action_token,
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
        )
        self.calls.append((compute_flow, compute_action_token))
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
    ("stage", "expected_scalar", "expected_per_sample", "expected_call"),
    [
        ("vlm_only", 1.5, torch.tensor([3.0, 0.0]), (False, True)),
        ("action_expert_only", 4.0, torch.tensor([4.0, 4.0]), (True, False)),
        ("joint", 5.5, torch.tensor([7.0, 4.0]), (True, True)),
    ],
)
def test_policy_training_stages_select_actionmem_objectives(
    stage,
    expected_scalar,
    expected_per_sample,
    expected_call,
):
    policy = PI05ActionMemPolicy.__new__(PI05ActionMemPolicy)
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
    policy.prepare_action = lambda batch: batch[ACTION]
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 3, dtype=torch.bool),
        ACTION_TOKENS: torch.tensor([[7, 8, 9, 257023], [7, 8, 9, 0]]),
        ACTION_TOKEN_MASK: torch.tensor([[True, True, True, True], [True, True, True, False]]),
        ACTION: torch.zeros(2, 1, 2),
    }

    scalar_loss, scalar_metrics = policy.forward(batch)
    per_sample_loss, per_sample_metrics = policy.forward(batch, reduction="none")

    assert scalar_loss.item() == expected_scalar
    assert torch.equal(per_sample_loss, expected_per_sample)
    assert policy.model.calls == [expected_call, expected_call]
    assert per_sample_metrics["loss"] == scalar_metrics["loss"]


def test_pi05_suffix_uses_time_adarms_without_a_state_projection():
    model = PI05ActionMemPytorch.__new__(PI05ActionMemPytorch)
    nn.Module.__init__(model)
    model.config = type(
        "Config",
        (),
        {"chunk_size": 16, "min_period": 4e-3, "max_period": 4.0},
    )()
    model.action_in_proj = nn.Linear(2, 4)
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model._apply_checkpoint = lambda function, *args, **kwargs: function(*args, **kwargs)

    embeddings, masks, attention, adarms = model.embed_suffix(
        noisy_actions=torch.zeros(2, 16, 2),
        timestep=torch.tensor([0.25, 0.75]),
    )

    assert embeddings.shape == (2, 16, 4)
    assert masks.shape == (2, 16)
    assert attention.shape == (2, 16)
    assert adarms.shape == (2, 4)
    assert not hasattr(model, "state_proj")


class _RestrictedHead(nn.Module):
    def forward(self, hidden_states):
        logits = torch.zeros(hidden_states.shape[0], 10, device=hidden_states.device)
        logits[:, 1] = 100.0
        logits[:, 7] = 3.0
        return logits


class _DummyInferencePaliGemma(nn.Module):
    def __init__(self):
        super().__init__()
        language_model = type(
            "LanguageModel",
            (),
            {"config": type("Config", (), {"_attn_implementation": "eager"})()},
        )()
        self.paligemma = type(
            "PaliGemma",
            (),
            {
                "lm_head": _RestrictedHead(),
                "model": type("PaliGemmaModel", (), {"language_model": language_model})(),
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


def test_inference_generates_token_without_a_separate_state_input(monkeypatch):
    model = PI05ActionMemPytorch.__new__(PI05ActionMemPytorch)
    nn.Module.__init__(model)
    model.config = type(
        "Config",
        (),
        {"num_inference_steps": 1, "chunk_size": 2, "max_action_dim": 2, "rtc_config": None},
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
    decoded_actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model.decode_action_tokens = lambda tokens: decoded_actions
    monkeypatch.setattr(
        modeling_pi05_actionmem,
        "euler_integrate",
        lambda denoise_fn, noise, num_steps, **kwargs: noise,
    )

    output = model.sample_actions(
        images=[],
        img_masks=[],
        lang_tokens=torch.ones(1, 3, dtype=torch.long),
        lang_masks=torch.ones(1, 3, dtype=torch.bool),
        action_tokens=torch.tensor([[7, 8, 9, 0]]),
        action_token_masks=torch.tensor([[True, True, True, False]]),
    )

    assert torch.equal(output, decoded_actions)
    assert len(model.paligemma_with_expert.forward_inputs) == 2
    assert torch.equal(
        model.paligemma_with_expert.forward_inputs[1],
        torch.full((1, 1, 2), 7.0),
    )


@pytest.mark.parametrize("stage", ["vlm_only", "action_expert_only", "joint"])
def test_default_peft_targets_match_native_pi05_modules(stage):
    policy = PI05ActionMemPolicy.__new__(PI05ActionMemPolicy)
    policy.config = type("Config", (), {"training_stage": stage})()
    pattern = policy._get_default_peft_targets()["target_modules"]

    assert bool(re.fullmatch(pattern, "model.time_mlp_in")) is (stage != "vlm_only")
    assert bool(re.fullmatch(pattern, "model.state_proj")) is False
