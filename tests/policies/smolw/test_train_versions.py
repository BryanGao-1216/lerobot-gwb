"""A/B/C/D equivalence and information-flow checks using real expert attention.

Only pretrained model construction and video encoding are replaced. The tests
execute SmolVLA's actual self/cross-attention, RoPE, cache, losses and sampler.
"""

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.smolvla import modeling_smolvla, processor_smolvla
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.smolw.configuration_smolw import SmolWConfig
from lerobot.policies.smolw.modeling_smolw import SmolWPolicy
from lerobot.policies.smolw.modeling_smolw_ablation import SmolWAblationFlowMatching
from lerobot.policies.smolw.processor_smolw import (
    SmolWStationaryActionPaddingProcessorStep,
    make_smolw_pre_post_processors,
    reconcile_smolw_processors,
)
from lerobot.processor import IdentityProcessorStep
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

CAMERA = "observation.images.image"


class _TinyVLMWithExpert(SmolVLMWithExpertModel):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        text_config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        self.config = SimpleNamespace(text_config=text_config)
        self.vlm = nn.Module()
        self.vlm.device = torch.device("cpu")
        self.vlm.config = self.config
        self.vlm.model = nn.Module()
        self.vlm.model.text_model = LlamaModel(text_config)
        self.vlm.model.vision_model = nn.Linear(3, 16)
        self.lm_expert = LlamaModel(copy.deepcopy(text_config))
        self.lm_expert.embed_tokens = None
        self.num_vlm_layers = self.num_expert_layers = 4
        self.self_attn_every_n_layers = 2
        self.attention_mode = kwargs["attention_mode"]
        if "cross" in self.attention_mode:
            for layer in self.lm_expert.layers[1::2]:
                layer.self_attn.k_proj = nn.Linear(8, 8, bias=False)
                layer.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        self.expert_hidden_size = 16
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.freeze_vision_encoder = kwargs["freeze_vision_encoder"]
        self.train_expert_only = kwargs["train_expert_only"]
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(fake_image_token_id=0, global_image_token_id=1)
        )
        self.set_requires_grad()

    def embed_image(self, image):
        return self.get_vlm_model().vision_model(image.mean(dim=(-1, -2))).unsqueeze(1)


class _MotionExtractor:
    def __init__(self):
        self.calls = []

    def encode(self, frames):
        self.calls.append(tuple(frames.shape))
        indices = torch.linspace(0, frames.shape[1] - 1, 16).long()
        return frames.index_select(1, indices).mean(dim=(-1, -2))[:, :, :2].flatten(1).detach()

    def encode_pair(self, past, future):
        self.calls.append("pair")
        return self.encode(past), self.encode(future)


def _config(version, **overrides):
    values = {
        "chunk_size": 3,
        "n_action_steps": 2,
        "max_state_dim": 4,
        "max_action_dim": 4,
        "device": "cpu",
        "resize_imgs_with_padding": (4, 4),
        "num_steps": 3,
        "train_expert_only": False,
        "input_features": {
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    }
    values.update(overrides)
    if version == "original":
        return SmolVLAConfig(**values)
    return SmolWConfig(train_version=version, motion_latent_dim=32, motion_projector_hidden_dim=16, **values)


@pytest.fixture
def make_policy(monkeypatch):
    monkeypatch.setattr(modeling_smolvla, "SmolVLMWithExpertModel", _TinyVLMWithExpert)

    def make(version, **overrides):
        torch.manual_seed(7)
        config = _config(version, **overrides)
        if version == "original":
            return SmolVLAPolicy(config)
        return SmolWPolicy(config, motion_extractor=_MotionExtractor())

    return make


def _batch(config):
    batch_size, temporal = 2, len(config.observation_delta_indices)
    return {
        CAMERA: torch.rand(batch_size, temporal, 3, 4, 4),
        OBS_STATE: torch.randn(batch_size, temporal, 3),
        ACTION: torch.randn(batch_size, config.chunk_size, 2),
        "action_is_pad": torch.tensor([[False, False, True], [False, True, True]]),
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 4], [5, 6, 0]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, True], [True, True, False]]),
    }


def _model_inputs():
    return {
        "images": [torch.rand(2, 3, 4, 4)],
        "img_masks": [torch.tensor([True, True])],
        "lang_tokens": torch.tensor([[2, 3, 4], [5, 6, 0]]),
        "lang_masks": torch.tensor([[True, True, True], [True, True, False]]),
        "state": torch.randn(2, 4),
        "past_motion": torch.randn(2, 32),
        "actions": torch.randn(2, 3, 4),
        "future_motion": torch.randn(2, 32),
        "noise": torch.randn(2, 3, 4),
        "z_noise": torch.randn(2, 16, 2),
        "time": torch.tensor([0.3, 0.7]),
    }


@pytest.mark.parametrize(
    "version, offsets",
    [("A", [0]), ("B", [-2, -1, 0]), ("C", [-2, -1, 0, 1, 2, 3]), ("D", [-2, -1, 0, 1, 2, 3])],
)
def test_version_controls_observations_and_roundtrips(version, offsets, tmp_path):
    config = _config(version)
    assert config.observation_delta_indices == offsets
    assert config.z_loss_weight == 1.0
    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.train_version == version
    assert loaded.observation_delta_indices == offsets


def test_invalid_version_is_rejected():
    with pytest.raises(ValueError, match="train_version"):
        _config("E")


@pytest.mark.parametrize("version", list("ABCD"))
def test_all_versions_preserve_original_preprocessing_and_action_padding(version, monkeypatch):
    monkeypatch.setattr(processor_smolvla, "TokenizerProcessorStep", lambda **kwargs: IdentityProcessorStep())
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.tensor([0.3, -0.2]), "std": torch.tensor([0.5, 0.7])},
    }
    config = _config(version)
    baseline, _ = processor_smolvla.make_smolvla_pre_post_processors(config, stats)
    pre, post = make_smolw_pre_post_processors(config, stats)
    pre.steps.insert(-1, SmolWStationaryActionPaddingProcessorStep(hold_dims=[-1]))
    pre, _ = reconcile_smolw_processors(config, pre, post)
    batch = _batch(config)
    batch["task"] = ["pick", "pick"]
    expected = baseline(copy.deepcopy(batch))
    actual = pre(copy.deepcopy(batch))
    assert [type(step) for step in pre.steps] == [type(step) for step in baseline.steps]
    torch.testing.assert_close(actual[ACTION], expected[ACTION], rtol=0, atol=0)
    assert torch.equal(actual["action_is_pad"], batch["action_is_pad"])


@pytest.mark.parametrize("attention_mode", ["cross_attn", "self_attn"])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_a_matches_original_parameters_loss_and_gradients(make_policy, attention_mode, reduction):
    original = make_policy("original", attention_mode=attention_mode)
    a = make_policy("A", attention_mode=attention_mode)
    assert type(a.model) is VLAFlowMatching
    assert not hasattr(a, "motion_extractor")
    assert a.state_dict().keys() == original.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, a.state_dict()[name], rtol=0, atol=0)
    batch = _batch(a.config)
    noise, time = torch.randn(2, 3, 4), torch.tensor([0.3, 0.7])
    expected, _ = original(copy.deepcopy(batch), noise=noise, time=time, reduction=reduction)
    actual, metrics = a(copy.deepcopy(batch), noise=noise, time=time, reduction=reduction)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "z_flow_loss" not in metrics
    actual.mean().backward()
    expected.mean().backward()
    for name, parameter in original.named_parameters():
        other = dict(a.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


def test_a_matches_original_action_queue_and_reset(make_policy):
    original, a = make_policy("original"), make_policy("A")
    for _ in range(7):
        batch = _batch(a.config)
        batch[CAMERA] = batch[CAMERA][:, 0]
        batch[OBS_STATE] = batch[OBS_STATE][:, 0]
        noise = torch.randn(2, 3, 4)
        expected = original.select_action(copy.deepcopy(batch), noise=noise)
        actual = a.select_action(copy.deepcopy(batch), noise=noise)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a.reset()
    assert len(a._queues[ACTION]) == 0
    assert not hasattr(a, "_motion_history")


def test_a_does_not_require_unused_motion_settings(make_policy):
    policy = make_policy(
        "A",
        train_expert_only=True,
        use_cache=False,
        motion_horizon=-1,
        memory_stride=0,
        vidtwin_num_frames=0,
        vidtwin_dtype="unused",
        z_loss_weight=0,
    )
    assert type(policy.model) is VLAFlowMatching
    assert policy.config.observation_delta_indices == [0]
    assert not hasattr(policy, "_motion_history")
    assert not hasattr(policy, "motion_extractor")


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize(
    "expert_only, freeze_vision, train_state",
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_a_matches_original_training_updates_rng_and_optimizer_resume(
    make_policy, bf16, expert_only, freeze_vision, train_state
):
    from accelerate import Accelerator

    from lerobot.optim.factory import make_optimizer_and_scheduler

    options = {
        "train_expert_only": expert_only,
        "freeze_vision_encoder": freeze_vision,
        "train_state_proj": train_state,
        "scheduler_warmup_steps": 2,
        "scheduler_decay_steps": 6,
    }
    original = make_policy("original", **options)
    original_rng = torch.get_rng_state().clone()
    a = make_policy("A", **options)
    assert torch.equal(original_rng, torch.get_rng_state())
    policies = [original, a]
    accelerator = Accelerator(cpu=True, mixed_precision="no")

    def optimizer_for(policy):
        cfg = SimpleNamespace(
            use_policy_training_preset=True,
            steps=6,
            optimizer=policy.config.get_optimizer_preset(),
            scheduler=policy.config.get_scheduler_preset(),
        )
        return make_optimizer_and_scheduler(cfg, policy)

    optimizers, schedulers = zip(*(optimizer_for(policy) for policy in policies), strict=True)
    for step in range(4):
        batch = _batch(a.config)
        rng = torch.get_rng_state().clone()
        rng_after = []
        metrics = []
        for policy, optimizer, scheduler in zip(policies, optimizers, schedulers, strict=True):
            torch.set_rng_state(rng)
            policy.train()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                loss, _ = policy(copy.deepcopy(batch))
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            metrics.append((loss.item(), grad_norm.item(), optimizer.param_groups[0]["lr"]))
            rng_after.append(torch.get_rng_state().clone())
        assert torch.equal(*rng_after)
        assert metrics[0] == metrics[1]
        for name, parameter in original.named_parameters():
            other = dict(a.named_parameters())[name]
            assert parameter.requires_grad == other.requires_grad
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
            left, right = optimizers[0].state.get(parameter, {}), optimizers[1].state.get(other, {})
            assert left.keys() == right.keys()
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        assert {name: module.training for name, module in original.named_modules()} == {
            name: module.training for name, module in a.named_modules()
        }
        if step == 1:
            # Exercise each policy's own saved optimizer layout. A retains the
            # trainable-only groups used by the user's existing A checkpoints.
            for index, policy in enumerate(policies):
                resumed_optimizer, resumed_scheduler = optimizer_for(policy)
                resumed_optimizer.load_state_dict(copy.deepcopy(optimizers[index].state_dict()))
                resumed_scheduler.load_state_dict(copy.deepcopy(schedulers[index].state_dict()))
                optimizers = tuple(
                    resumed_optimizer if i == index else value for i, value in enumerate(optimizers)
                )
                schedulers = tuple(
                    resumed_scheduler if i == index else value for i, value in enumerate(schedulers)
                )
    # Stochastic inference consumes exactly the same RNG, including queued steps.
    original.reset()
    a.reset()
    for _ in range(5):
        batch = _batch(a.config)
        rng = torch.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            expected = original.select_action(copy.deepcopy(batch))
            expected_rng = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            actual = a.select_action(copy.deepcopy(batch))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("padding", ["none", "all", "mixed"])
def test_a_matches_original_with_multiple_cameras_special_tokens_and_padding(make_policy, padding):
    features = _config("original").input_features
    features["observation.images.wrist"] = copy.deepcopy(features[CAMERA])
    features["observation.images.missing"] = copy.deepcopy(features[CAMERA])
    options = {"input_features": features, "add_image_special_tokens": True, "prefix_length": 24}
    original, a = (make_policy(version, **copy.deepcopy(options)) for version in ["original", "A"])
    batch = _batch(a.config)
    batch["observation.images.wrist"] = torch.rand(2, 1, 3, 3, 5)
    batch["observation.images.wrist_padding_mask"] = torch.tensor([[True], [False]])
    if padding == "none":
        batch.pop("action_is_pad")
    elif padding == "all":
        batch["action_is_pad"].fill_(True)
    rng = torch.get_rng_state().clone()
    expected, _ = original(copy.deepcopy(batch))
    torch.set_rng_state(rng)
    actual, _ = a(copy.deepcopy(batch))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    noise = torch.randn(2, 3, 4)
    torch.testing.assert_close(
        a.predict_action_chunk(copy.deepcopy(batch), noise=noise),
        original.predict_action_chunk(copy.deepcopy(batch), noise=noise),
        rtol=0,
        atol=0,
    )


def test_converted_base_preserves_every_original_weight(make_policy):
    from lerobot.policies.smolw.convert_smolvla_checkpoint import _copy_compatible_weights

    original, legacy = make_policy("original"), make_policy(None)
    with torch.no_grad():
        original.model.action_out_proj.weight.add_(1)
    _copy_compatible_weights(original, legacy)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)
    legacy.model.action_out_proj = nn.Linear(16, 5)
    with pytest.raises(RuntimeError, match="incompatible keys"):
        _copy_compatible_weights(original, legacy)


def test_a_rejects_incomplete_base_even_with_default_non_strict_loading(make_policy, tmp_path):
    from safetensors.torch import load_file, save_file

    legacy = make_policy(None)
    legacy.save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    weights.pop("model.action_out_proj.weight")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="missing baseline SmolVLA weights"):
        SmolWPolicy.from_pretrained(tmp_path, config=_config("A"))


@pytest.mark.parametrize("corruption", [None, "weight", "dtype", "normalizer", "processor"])
def test_base_artifact_audit_detects_weights_and_processor_changes(
    make_policy, monkeypatch, tmp_path, corruption
):
    from safetensors.torch import load_file, save_file

    from lerobot.policies.smolw.audit_smolvla_base import audit
    from lerobot.policies.smolw.convert_smolvla_checkpoint import _copy_compatible_weights
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    monkeypatch.setattr(processor_smolvla, "TokenizerProcessorStep", lambda **kwargs: IdentityProcessorStep())
    original, legacy = make_policy("original"), make_policy(None)
    _copy_compatible_weights(original, legacy)
    source, target = tmp_path / "source", tmp_path / "target"
    original.save_pretrained(source)
    legacy.save_pretrained(target)
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.zeros(2), "std": torch.ones(2)},
    }
    pre, post = processor_smolvla.make_smolvla_pre_post_processors(original.config, stats)
    for directory in [source, target]:
        if directory == target:
            pre, post = reconcile_smolw_processors(legacy.config, pre, post)
        pre.save_pretrained(directory, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
        post.save_pretrained(directory, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
    if corruption in {"weight", "dtype"}:
        path = target / "model.safetensors"
        weights = load_file(path)
        key = "model.action_out_proj.weight"
        weights[key] = weights[key] + 1 if corruption == "weight" else weights[key].to(torch.bfloat16)
        save_file(weights, path)
    elif corruption in {"normalizer", "processor"}:
        path = target / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        config = json.loads(path.read_text())
        if corruption == "processor":
            config["steps"].pop(0)
            path.write_text(json.dumps(config))
        else:
            state_path = target / next(step["state_file"] for step in config["steps"] if "state_file" in step)
            state = load_file(state_path)
            key = next(iter(state))
            state[key] = state[key] + 1
            save_file(state, state_path)
    result = audit(source, target)
    assert result["weights_and_processors_match"] == (corruption is None)
    assert bool(result["issues"]) == (corruption is not None)


def test_a_reloads_converted_processors_with_original_stats_and_postprocessing(monkeypatch, tmp_path):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    monkeypatch.setattr(processor_smolvla, "TokenizerProcessorStep", lambda **kwargs: IdentityProcessorStep())
    stats = {
        OBS_STATE: {"mean": torch.tensor([1.0, 2.0, 3.0]), "std": torch.tensor([0.2, 0.4, 0.8])},
        ACTION: {"mean": torch.tensor([0.3, -0.2]), "std": torch.tensor([0.5, 0.7])},
    }
    original_pre, original_post = make_pre_post_processors(_config("original"), dataset_stats=stats)
    pre, post = make_pre_post_processors(_config("original"), dataset_stats=stats)
    pre, post = reconcile_smolw_processors(_config(None), pre, post)
    pre.save_pretrained(tmp_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    post.save_pretrained(tmp_path, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
    loaded_pre, loaded_post = make_pre_post_processors(_config("A"), pretrained_path=str(tmp_path))
    assert loaded_pre.get_config() == original_pre.get_config()
    assert loaded_post.get_config() == original_post.get_config()
    batch = {**_batch(_config("A")), "task": ["pick", "place"]}
    actual, expected = loaded_pre(copy.deepcopy(batch)), original_pre(copy.deepcopy(batch))
    for key in batch:
        if isinstance(expected[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]
    action = torch.randn(2, 2)
    torch.testing.assert_close(loaded_post(action), original_post(action), rtol=0, atol=0)


@pytest.mark.parametrize("attention_mode", ["cross_attn", "self_attn"])
def test_c_adds_z_supervision_without_changing_b_action_path(make_policy, attention_mode):
    b, c = (make_policy(v, attention_mode=attention_mode).model for v in "BC")
    inputs = _model_inputs()
    b_out, c_out = b(**inputs), c(**inputs)
    torch.testing.assert_close(b_out["flow_losses"], c_out["flow_losses"], rtol=1e-5, atol=1e-6)
    assert "z_flow_losses" not in b_out
    assert not hasattr(b, "z_token_in_proj")
    c_out["flow_losses"].mean().backward()
    assert c.past_motion_projector[1].weight.grad.norm() > 0
    assert torch.count_nonzero(c.z_token_in_proj.weight.grad) == 0
    c.zero_grad()
    c_out = c(**{**inputs, "future_motion": torch.randn(2, 32), "z_noise": torch.randn(2, 16, 2)})
    torch.testing.assert_close(b_out["flow_losses"], c_out["flow_losses"], rtol=1e-5, atol=1e-6)
    c_out["z_flow_losses"].mean().backward()
    assert c.z_token_in_proj.weight.grad.norm() > 0
    assert c.past_motion_projector[1].weight.grad.norm() > 0


@pytest.mark.parametrize("attention_mode", ["cross_attn", "self_attn"])
def test_d_zero_gate_matches_c_and_learns_to_read_z(make_policy, attention_mode):
    c, d = (make_policy(v, attention_mode=attention_mode).model for v in "CD")
    inputs = _model_inputs()
    c_out, d_out = c(**inputs), d(**inputs)
    for key in c_out:
        torch.testing.assert_close(c_out[key], d_out[key], rtol=0, atol=0)
    d_out["flow_losses"].mean().backward()
    assert d.motion_condition_gate.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(d.motion_condition_attention.in_proj_weight.grad) == 0
    d.zero_grad()
    with torch.no_grad():
        d.motion_condition_gate.weight.fill_(0.5)
    opened = d(**inputs)
    changed = d(**{**inputs, "z_noise": inputs["z_noise"] + torch.randn_like(inputs["z_noise"])})
    assert not torch.allclose(opened["flow_losses"], changed["flow_losses"])
    changed["flow_losses"].mean().backward()
    assert d.motion_condition_attention.in_proj_weight.grad.norm() > 0
    assert d.z_token_in_proj.weight.grad.norm() > 0


def test_b_c_d_sampling_and_cached_training_agree(make_policy):
    models = [make_policy(v).model for v in "BCD"]
    inputs = _model_inputs()
    sample_inputs = {k: v for k, v in inputs.items() if k not in {"actions", "future_motion", "time"}}
    outputs = [model.sample_actions(**sample_inputs) for model in models]
    for result in outputs[1:]:
        torch.testing.assert_close(result, outputs[0], rtol=1e-5, atol=1e-6)
    # D exercises all z/action attention rows in both full and cached execution.
    d = models[-1]
    with torch.no_grad():
        d.motion_condition_gate.weight.fill_(0.5)
    prefix_pad, cache = d.run_condition_prefix(
        inputs["images"],
        inputs["img_masks"],
        inputs["lang_tokens"],
        inputs["lang_masks"],
        inputs["state"],
        inputs["past_motion"],
    )
    t = inputs["time"][:, None, None]
    z_target = d.motion_to_z(inputs["future_motion"])
    suffix, attention, positions = d._embed_ablation_suffix(
        prefix_pad,
        t * inputs["noise"] + (1 - t) * inputs["actions"],
        inputs["time"],
        t * inputs["z_noise"] + (1 - t) * z_target,
    )
    (_, hidden), _ = d.vlm_with_expert.forward(
        inputs_embeds=[None, suffix],
        attention_mask=attention,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
        fill_kv_cache=False,
    )
    action_v, _ = d._project_ablation_output(hidden, has_z=True)
    cached_loss = (inputs["noise"] - inputs["actions"] - action_v).square()
    torch.testing.assert_close(cached_loss, d(**inputs)["flow_losses"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("version", list("ABCD"))
def test_policy_training_inference_and_checkpoint_roundtrip(make_policy, version, tmp_path):
    policy = make_policy(version)
    batch = _batch(policy.config)
    loss, metrics = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    if version == "B":
        assert metrics["z_flow_loss"] == 0
        assert "pair" not in policy.motion_extractor.calls
    elif version in "CD":
        assert "pair" in policy.motion_extractor.calls
        assert metrics["loss"] == pytest.approx(metrics["action_flow_loss"] + metrics["weighted_z_flow_loss"])
    current = {
        **batch,
        CAMERA: batch[CAMERA][:, policy.config.current_observation_position],
        OBS_STATE: batch[OBS_STATE][:, policy.config.current_observation_position],
    }
    policy.reset()
    for _ in range(3):
        assert policy.select_action(current).shape == (2, 2)
    if version != "A":
        assert len(policy._motion_history) == 3  # includes queue-consumption steps
        if version == "D":
            with torch.no_grad():
                policy.model.motion_condition_gate.weight.fill_(0.4)
    policy.save_pretrained(tmp_path)
    loaded = SmolWPolicy.from_pretrained(tmp_path, motion_extractor=_MotionExtractor(), strict=True)
    assert loaded.config.train_version == version
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)


def test_ablation_mask_preserves_history_and_action_positions():
    prefix = torch.tensor([[True, False, True]])  # final position is M_t
    actions = torch.ones(1, 3, dtype=torch.bool)
    z = torch.ones(1, 16, dtype=torch.bool)
    b_mask, b_pos = SmolWAblationFlowMatching.make_ablation_attention(prefix, actions, actions)
    c_mask, c_pos = SmolWAblationFlowMatching.make_ablation_attention(prefix, actions, actions, z)
    assert c_mask[:, -3:, 2].all()
    assert not c_mask[:, -3:, 3:19].any()
    assert not c_mask[:, :16, -3:].any()
    torch.testing.assert_close(c_pos[:, -3:], b_pos)
    assert torch.equal(torch.cat([c_mask[:, -3:, :3], c_mask[:, -3:, -3:]], dim=2), b_mask)


@pytest.mark.parametrize("version", list("ABCD"))
def test_existing_base_loads_with_launch_version_override(make_policy, version, tmp_path):
    legacy = make_policy(None)
    legacy.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    serialized = json.loads(config_path.read_text())
    serialized.pop("train_version")  # actual pre-ablation checkpoint format
    config_path.write_text(json.dumps(serialized))
    old_config = PreTrainedConfig.from_pretrained(tmp_path)
    assert old_config.train_version is None
    config = PreTrainedConfig.from_pretrained(tmp_path, cli_overrides=[f"--train_version={version}"])
    loaded = SmolWPolicy.from_pretrained(tmp_path, config=config, motion_extractor=_MotionExtractor())
    assert loaded.config.train_version == version
    legacy_state = legacy.state_dict()
    for name, value in loaded.state_dict().items():
        if name in legacy_state:
            torch.testing.assert_close(value, legacy_state[name], rtol=0, atol=0)
    if version == "D":
        assert loaded.model.motion_condition_gate.weight.item() == 0


@pytest.mark.parametrize("version", list("ABCD"))
def test_bfloat16_autocast_training_and_inference(make_policy, version):
    policy = make_policy(version)
    if version == "D":
        with torch.no_grad():
            policy.model.motion_condition_gate.weight.fill_(0.2)
    batch = _batch(policy.config)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in policy.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actions = policy.predict_action_chunk(batch)
    assert actions.shape == (2, 3, 2)
    assert torch.isfinite(actions).all()


@pytest.mark.parametrize("version", ["A", "B", "C", "D", "invalid"])
def test_launch_script_selects_version_without_changing_hyperparameters(version, tmp_path):
    script = Path(__file__).resolve().parents[3] / "train_smolw_lr.sh"
    for executable, content in {
        "accelerate": '#!/bin/sh\nprintf "%s\\n" "$@"\n',
        "lerobot-train": "#!/bin/sh\nexit 0\n",
    }.items():
        path = tmp_path / executable
        path.write_text(content)
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin", "train_version": version}
    for key in (
        "HORIZON",
        "N_ACTION_STEPS",
        "MEMORY_STRIDE",
        "BATCH_SIZE",
        "OUTPUT_DIR",
        "TENSORBOARD_LOG_DIR",
    ):
        env.pop(key, None)
    result = subprocess.run(["/bin/sh", str(script)], env=env, capture_output=True, text=True)
    if version == "invalid":
        assert result.returncode != 0
        assert "train_version must be" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    for expected in (
        f"--policy.train_version={version}",
        "--policy.z_loss_weight=1.0",
        "--policy.train_expert_only=false",
        "--policy.chunk_size=16",
        "--policy.n_action_steps=10",
        "--batch_size=64",
        "--steps=100000",
        "--policy.optimizer_lr=3e-5",
        f"--output_dir=/data1/gaowenbing/WorkSpace/models/smolw-union-{version}",
    ):
        assert expected in args


@pytest.mark.parametrize("version", ["A", "B", "C", "D", "invalid"])
def test_eval_script_selects_matching_checkpoint_without_reinterpreting_version(version, tmp_path):
    script = Path(__file__).resolve().parents[3] / "test_smolw.sh"
    executable = tmp_path / "lerobot-eval"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:/usr/bin:/bin", "train_version": version}
    for key in ("POLICY_PATH", "OUTPUT_DIR", "CHECKPOINT_STEP"):
        env.pop(key, None)
    result = subprocess.run(["/bin/sh", str(script)], env=env, capture_output=True, text=True)
    if version == "invalid":
        assert result.returncode != 0
        return
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert (
        f"--policy.path=/data1/gaowenbing/WorkSpace/models/smolw-union-{version}/checkpoints/030000/pretrained_model"
    ) in args
    assert not any(arg.startswith("--policy.train_version=") for arg in args)
    assert "--policy.n_action_steps=10" in args
    assert "--eval.n_episodes=5" in args
    env["POLICY_PATH"] = "/custom/legacy-checkpoint"
    result = subprocess.run(["/bin/sh", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--policy.path=/custom/legacy-checkpoint" in result.stdout.splitlines()
