from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.smolw.configuration_smolw import SmolWConfig
from lerobot.policies.smolw.modeling_smolw import SmolWFlowMatching, SmolWPolicy
from lerobot.policies.smolw.vidtwin_motion_encoder import (
    _BUNDLED_VIDTWIN_CONFIG,
    VidTwinMotionExtractor,
    _get_obj_from_str,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

CAMERA = "observation.images.image"
HORIZON = 16


def test_policy_is_independently_registered():
    config = make_policy_config(
        "smolw",
        chunk_size=HORIZON,
        n_action_steps=HORIZON,
        motion_horizon=HORIZON,
        device="cpu",
    )

    assert isinstance(config, SmolWConfig)
    assert get_policy_class("smolw") is SmolWPolicy


def test_config_builds_past_and_future_lerobot_timestamps():
    config = SmolWConfig(
        chunk_size=HORIZON,
        n_action_steps=HORIZON,
        motion_horizon=HORIZON,
        memory_stride=2,
        device="cpu",
    )

    assert config.past_motion_delta_indices == list(range(-30, 1, 2))
    assert config.future_motion_delta_indices == list(range(1, HORIZON + 1))
    assert config.observation_delta_indices == list(range(-30, 1, 2)) + list(range(1, HORIZON + 1))
    assert config.past_motion_positions == list(range(HORIZON))
    assert config.future_motion_positions == list(range(HORIZON, 2 * HORIZON))
    assert config.current_observation_position == HORIZON - 1
    assert config.drop_n_last_frames == HORIZON
    assert config.motion_token_dim == 112
    assert config.train_mode == "motion_only"
    assert config.training_stage is None
    assert not config.train_expert_only
    assert not config.vidtwin_sample_posterior
    assert not hasattr(config, "vidtwin_repo_path")
    assert not hasattr(config, "vidtwin_config_path")
    assert not config.tensorboard_enable
    assert config.tensorboard_log_freq == 100
    assert config.z_condition_warmup_steps == 0


@pytest.mark.parametrize("train_mode", ["motion_only", "action_only", "jointly"])
def test_all_train_modes_request_future_motion_target(train_mode):
    config = SmolWConfig(
        chunk_size=HORIZON,
        n_action_steps=HORIZON,
        motion_horizon=HORIZON,
        memory_stride=2,
        train_mode=train_mode,
        device="cpu",
    )

    assert config.observation_delta_indices == list(range(-30, 1, 2)) + list(range(1, HORIZON + 1))
    assert config.past_motion_positions == list(range(HORIZON))
    assert config.future_motion_positions == list(range(HORIZON, 2 * HORIZON))
    assert config.drop_n_last_frames == HORIZON


def test_config_rejects_unknown_train_mode():
    with pytest.raises(ValueError, match="train_mode"):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON,
            train_mode="unknown",
            device="cpu",
        )


def test_config_enforces_fixed_vidtwin_frame_count():
    with pytest.raises(ValueError, match="vidtwin_num_frames=16"):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON,
            vidtwin_num_frames=8,
            device="cpu",
        )


def test_config_can_be_loaded_with_train_mode_override(tmp_path):
    motion_config = SmolWConfig(
        chunk_size=HORIZON,
        n_action_steps=HORIZON,
        motion_horizon=HORIZON,
        train_mode="motion_only",
        device="cpu",
    )
    motion_config.save_pretrained(tmp_path)

    action_config = PreTrainedConfig.from_pretrained(
        tmp_path,
        cli_overrides=["--train_mode=action_only"],
    )

    assert isinstance(action_config, SmolWConfig)
    assert action_config.train_mode == "action_only"
    assert action_config.observation_delta_indices == list(range(-15, HORIZON + 1))
    assert action_config.drop_n_last_frames == HORIZON


def test_legacy_training_stage_is_migrated_to_train_mode():
    config = SmolWConfig(
        chunk_size=HORIZON,
        n_action_steps=HORIZON,
        motion_horizon=HORIZON,
        training_stage="action_expert_only",
        device="cpu",
    )

    assert config.train_mode == "action_only"
    assert config.training_stage is None
    assert not config.train_expert_only


def test_z_loss_weight_must_be_non_negative():
    with pytest.raises(ValueError, match="z_loss_weight"):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON,
            z_loss_weight=-1.0,
            device="cpu",
        )


def test_z_condition_warmup_steps_must_be_non_negative():
    with pytest.raises(ValueError, match="z_condition_warmup_steps"):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON,
            z_condition_warmup_steps=-1,
            device="cpu",
        )


@pytest.mark.parametrize(
    "field",
    [
        "tensorboard_log_freq",
        "tensorboard_flush_secs",
        "tensorboard_max_queue",
        "tensorboard_histogram_freq",
    ],
)
def test_tensorboard_intervals_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON,
            device="cpu",
            **{field: 0},
        )


def test_vidtwin_architecture_config_is_bundled_and_uses_internal_targets():
    config_text = _BUNDLED_VIDTWIN_CONFIG.read_text(encoding="utf-8")

    assert "lerobot.policies.smolw.vidtwin.models.vidtwin_ae" in config_text
    assert "target: vidtwin." not in config_text


def test_bundled_qformer_runs_with_lerobot_transformers_version():
    omegaconf = pytest.importorskip("omegaconf")
    config = omegaconf.OmegaConf.load(_BUNDLED_VIDTWIN_CONFIG)
    qformer_config = config.model.params.temporal_qformer_config
    qformer = _get_obj_from_str(qformer_config.target)(**qformer_config.params)

    with torch.inference_mode():
        output = qformer(torch.randn(1, 8, qformer_config.params.encoder_hidden_size))

    assert output.shape == (
        1,
        qformer_config.params.num_query_tokens,
        qformer_config.params.query_hidden_size,
    )


def test_config_keeps_motion_and_action_horizons_aligned():
    with pytest.raises(ValueError, match="motion_horizon == chunk_size"):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON - 1,
            device="cpu",
        )


def test_action_horizon_can_differ_from_fixed_vidtwin_token_count():
    config = SmolWConfig(
        chunk_size=4,
        n_action_steps=4,
        motion_horizon=4,
        device="cpu",
    )

    assert config.motion_horizon == 4
    assert config.vidtwin_num_frames == 16
    assert config.motion_token_dim == 112


def test_flatten_motion_latents_matches_cowvla_order():
    z_motion_x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    z_motion_y = torch.arange(10, 14, dtype=torch.float32).reshape(1, 1, 2, 2)

    flattened = VidTwinMotionExtractor.flatten_motion_latents(z_motion_x, z_motion_y)

    assert torch.equal(flattened, torch.tensor([[0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0]]))


def test_vidtwin_preprocess_samples_fixed_frame_count_and_normalizes(tmp_path):
    extractor = VidTwinMotionExtractor(
        checkpoint_path=tmp_path / "model.ckpt",
        num_frames=4,
        input_height=2,
        input_width=2,
        dtype="float32",
        expected_latent_dim=8,
    )
    frames = torch.stack(
        [torch.zeros(3, 2, 4), torch.ones(3, 2, 4)],
        dim=0,
    ).unsqueeze(0)

    video = extractor.preprocess(frames)

    assert video.shape == (1, 3, 4, 2, 2)
    assert torch.equal(video[:, :, 0], torch.full((1, 3, 2, 2), -1.0))
    assert torch.equal(video[:, :, -1], torch.full((1, 3, 2, 2), 1.0))


def _temporal_policy() -> SmolWPolicy:
    policy = SmolWPolicy.__new__(SmolWPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features={CAMERA: object()},
        observation_delta_indices=[-6, -4, -2, 0, 1, 2, 3, 4],
        current_observation_position=3,
        past_motion_positions=[0, 1, 2, 3],
        future_motion_positions=[4, 5, 6, 7],
        motion_horizon=4,
        memory_stride=2,
    )
    policy.motion_camera_key = CAMERA
    return policy


def test_current_observation_does_not_read_future_frames():
    policy = _temporal_policy()
    images = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)
    states = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    batch = {
        CAMERA: images,
        f"{CAMERA}_is_pad": torch.tensor([[True, False, False, False, False, False, False, False]]),
        OBS_STATE: states,
    }

    current = policy._current_batch(batch)

    assert torch.equal(current[CAMERA], images[:, 3])
    assert torch.equal(current[OBS_STATE], states[:, 3])
    assert torch.equal(current[f"{CAMERA}_padding_mask"], torch.tensor([True]))


def test_motion_clips_use_strided_past_and_contiguous_future():
    policy = _temporal_policy()
    frames = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)

    past, future = policy.prepare_motion_clips({CAMERA: frames})

    assert torch.equal(past.flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(future.flatten(), torch.tensor([4.0, 5.0, 6.0, 7.0]))


def test_inference_history_repeats_episode_start_and_respects_stride():
    policy = _temporal_policy()
    policy.config.motion_horizon = 3
    policy.config.memory_stride = 2
    policy._motion_history = deque(maxlen=5)
    for value in range(5):
        policy._motion_history.append(torch.tensor([[[[float(value)]]]]))

    clip = policy._history_motion_clip()

    assert torch.equal(clip.flatten(), torch.tensor([0.0, 2.0, 4.0]))
    policy._motion_history = deque([torch.tensor([[[[9.0]]]])], maxlen=5)
    assert torch.equal(policy._history_motion_clip().flatten(), torch.tensor([9.0, 9.0, 9.0]))


class _DummyVLM:
    @staticmethod
    def embed_image(image):
        return image

    @staticmethod
    def embed_language_tokens(tokens):
        return tokens.float().unsqueeze(-1).expand(-1, -1, 4)


class _DummyVLMWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))
        self.expert_hidden_size = 4

    @staticmethod
    def embed_image(image):
        return image

    @staticmethod
    def embed_language_tokens(tokens):
        return tokens.float().unsqueeze(-1).expand(-1, -1, 4)

    def forward(self, *, inputs_embeds, fill_kv_cache, **kwargs):
        del kwargs
        prefix, suffix = inputs_embeds
        if fill_kv_cache:
            return (prefix, None), {"prefix": prefix}
        return (None, suffix), None


class _StageVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.vision_model = nn.Linear(2, 2)
        self.model.connector = nn.Linear(2, 2)
        self.model.text_model = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2, 2)


class _StageVLMWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = _StageVLM()
        self.lm_expert = nn.Linear(2, 2)

    def get_vlm_model(self):
        return self.vlm.model


def _mode_model(train_mode: str) -> SmolWFlowMatching:
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        train_mode=train_mode,
        freeze_vision_encoder=True,
        train_state_proj=True,
    )
    model.vlm_with_expert = _StageVLMWithExpert()
    model.state_proj = nn.Linear(2, 2)
    model.mt_query_embedding = nn.Embedding(1, 2)
    model.past_motion_projector = nn.Linear(2, 2)
    model.future_motion_head = nn.Linear(2, 2)
    model.z_token_in_proj = nn.Linear(2, 2)
    model.z_time_mlp_in = nn.Linear(2, 2)
    model.z_time_mlp_out = nn.Linear(2, 2)
    model.z_token_out_proj = nn.Linear(2, 2)
    model.action_in_proj = nn.Linear(2, 2)
    model.action_out_proj = nn.Linear(2, 2)
    model.action_time_mlp_in = nn.Linear(2, 2)
    model.action_time_mlp_out = nn.Linear(2, 2)
    model._train_mode_configured = False
    return model


def test_train_mode_freezes_unselected_branch_and_vision_encoder():
    motion_model = _mode_model("motion_only")
    # Simulate a tensor SmolVLA already froze to avoid DDP unused parameters.
    motion_model.vlm_with_expert.vlm.model.text_model.bias.requires_grad_(False)
    motion_model.configure_train_mode()

    assert motion_model.vlm_with_expert.vlm.model.text_model.weight.requires_grad
    assert not motion_model.vlm_with_expert.vlm.model.text_model.bias.requires_grad
    assert not motion_model.vlm_with_expert.vlm.model.vision_model.weight.requires_grad
    assert motion_model.vlm_with_expert.vlm.model.connector.weight.requires_grad
    assert motion_model.future_motion_head.weight.requires_grad
    assert not motion_model.vlm_with_expert.lm_expert.weight.requires_grad
    assert not motion_model.z_token_in_proj.weight.requires_grad
    assert not motion_model.z_token_out_proj.weight.requires_grad

    action_model = _mode_model("action_only")
    action_model.configure_train_mode()

    assert not action_model.vlm_with_expert.vlm.model.text_model.weight.requires_grad
    assert not action_model.future_motion_head.weight.requires_grad
    assert action_model.vlm_with_expert.lm_expert.weight.requires_grad
    assert action_model.state_proj.weight.requires_grad
    assert action_model.action_in_proj.weight.requires_grad
    assert action_model.z_token_in_proj.weight.requires_grad
    assert action_model.z_token_out_proj.weight.requires_grad

    action_model.train()
    assert not action_model.vlm_with_expert.vlm.training
    assert action_model.vlm_with_expert.lm_expert.training

    joint_model = _mode_model("jointly")
    joint_model.configure_train_mode()

    assert joint_model.vlm_with_expert.vlm.model.text_model.weight.requires_grad
    assert joint_model.future_motion_head.weight.requires_grad
    assert joint_model.vlm_with_expert.lm_expert.weight.requires_grad
    assert joint_model.z_token_in_proj.weight.requires_grad
    assert joint_model.z_token_out_proj.weight.requires_grad

    no_state_model = _mode_model("action_only")
    no_state_model.config.train_state_proj = False
    no_state_model.configure_train_mode()
    assert not no_state_model.state_proj.weight.requires_grad


def test_mt_query_is_appended_after_original_smolvla_prefix():
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(motion_latent_dim=3)
    model.vlm_with_expert = _DummyVLM()
    model.add_image_special_tokens = False
    model.prefix_length = -1
    model.state_proj = nn.Linear(2, 4, bias=False)
    model.mt_query_embedding = nn.Embedding(1, 4)
    model.past_motion_projector = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 4))

    embeddings, padding_masks, attention_masks = model.embed_prefix_with_motion(
        images=[torch.ones(1, 2, 4)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[2, 3]]),
        lang_masks=torch.tensor([[True, True]]),
        state=torch.ones(1, 2),
        past_motion=torch.tensor([[1.0, 2.0, 3.0]]),
    )

    assert embeddings.shape == (1, 6, 4)
    assert padding_masks.shape == (1, 6)
    assert padding_masks.all()
    assert attention_masks[0, -1]
    expected_query = (
        model.mt_query_embedding.weight[0] + model.past_motion_projector(torch.tensor([[1.0, 2.0, 3.0]]))[0]
    )
    assert torch.allclose(embeddings[0, -1], expected_query)


def test_all_actions_see_all_z_while_z_never_sees_actions():
    prefix_masks = torch.tensor([[True, True, False, True]])
    z_masks = torch.tensor([[True, True, True]])
    action_masks = torch.tensor([[True, True]])
    action_attention = torch.ones(1, 2, dtype=torch.bool)

    original_attention, original_position_ids = SmolWFlowMatching.make_action_attention(
        prefix_masks,
        action_masks,
        action_attention,
    )
    attention, position_ids = SmolWFlowMatching.make_action_z_attention(
        prefix_masks,
        z_masks,
        action_masks,
        action_attention,
    )

    assert attention.shape == (1, 5, 9)
    assert torch.equal(
        attention[0],
        torch.tensor(
            [
                # Every z sees the full valid prefix and all z, but no action.
                [True, True, False, True, True, True, True, False, False],
                [True, True, False, True, True, True, True, False, False],
                [True, True, False, True, True, True, True, False, False],
                # Every action sees all z while retaining original a-to-a causality.
                [True, True, False, False, True, True, True, True, False],
                [True, True, False, False, True, True, True, True, True],
            ]
        ),
    )
    # Extract [prefix, action] columns and verify that
    # action-to-action behavior is byte-for-byte the original SmolVLA mask.
    original_columns = torch.tensor([0, 1, 2, 3, 7, 8])
    assert torch.equal(attention[:, 3:].index_select(2, original_columns), original_attention)
    assert torch.equal(position_ids[:, 3:], original_position_ids)
    assert torch.equal(position_ids, torch.tensor([[3, 4, 5, 2, 3]]))

    warmup_attention, warmup_position_ids = SmolWFlowMatching.make_action_z_attention(
        prefix_masks,
        z_masks,
        action_masks,
        action_attention,
        action_can_see_z=False,
    )
    # During warmup, only action->z edges are removed. The original action
    # mask, all z rows, and all position ids remain unchanged.
    assert not warmup_attention[:, 3:, 4:7].any()
    assert torch.equal(warmup_attention[:, :3], attention[:, :3])
    assert torch.equal(warmup_attention[:, 3:].index_select(2, original_columns), original_attention)
    assert torch.equal(warmup_position_ids, position_ids)


def test_motion_to_z_uses_fixed_per_slot_normalization():
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        motion_latent_dim=8,
        vidtwin_num_frames=2,
        motion_token_dim=4,
        detach_motion_condition=False,
        motion_condition_scale=2.0,
    )
    future_motion = torch.tensor([[1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0]])

    z_target = model.motion_to_z(future_motion)
    expected = torch.nn.functional.layer_norm(future_motion.reshape(1, 2, 4), (4,)) * 2.0

    assert z_target.shape == (1, 2, 4)
    assert torch.allclose(z_target, expected)
    assert torch.allclose(z_target.mean(dim=-1), torch.zeros(1, 2), atol=1e-6)


def test_motion_first_forward_and_sampling_keep_smolvla_action_shapes():
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        motion_latent_dim=4,
        motion_horizon=2,
        vidtwin_num_frames=2,
        motion_token_dim=2,
        motion_condition_scale=1.0,
        detach_motion_condition=False,
        z_condition_warmup_steps=1,
        chunk_size=2,
        max_action_dim=2,
        min_period=4e-3,
        max_period=4.0,
        num_steps=2,
        rtc_config=None,
    )
    model.vlm_with_expert = _DummyVLMWithExpert()
    model.add_image_special_tokens = False
    model.prefix_length = -1
    model.state_proj = nn.Linear(2, 4)
    model.action_in_proj = nn.Linear(2, 4)
    model.action_time_mlp_in = nn.Linear(8, 4)
    model.action_time_mlp_out = nn.Linear(4, 4)
    model.action_out_proj = nn.Linear(4, 2)
    model.mt_query_embedding = nn.Embedding(1, 4)
    model.past_motion_projector = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
    model.future_motion_head = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
    model.z_token_in_proj = nn.Linear(2, 4)
    model.z_time_mlp_in = nn.Linear(8, 4)
    model.z_time_mlp_out = nn.Linear(4, 4)
    model.z_token_out_proj = nn.Linear(4, 2)
    model.register_buffer("z_condition_step", torch.zeros((), dtype=torch.long), persistent=True)
    model.rtc_processor = None

    inputs = {
        "images": [torch.ones(1, 2, 4)],
        "img_masks": [torch.tensor([True])],
        "lang_tokens": torch.tensor([[2, 3]]),
        "lang_masks": torch.tensor([[True, True]]),
        "state": torch.ones(1, 2),
        "past_motion": torch.ones(1, 4),
    }
    output = model.forward(
        **inputs,
        actions=torch.ones(1, 2, 2),
        future_motion_target=torch.zeros(1, 4),
        noise=torch.zeros(1, 2, 2),
        z_noise=torch.zeros(1, 2, 2),
        time=torch.full((1,), 0.5),
    )
    actions = model.sample_actions(
        **inputs,
        noise=torch.zeros(1, 2, 2),
        z_noise=torch.zeros(1, 2, 2),
    )

    assert output["flow_losses"].shape == (1, 2, 2)
    assert output["z_flow_losses"].shape == (1,)
    assert output["motion_losses"].shape == (1,)
    assert output["predicted_future_motion"].shape == (1, 4)
    assert torch.equal(output["z_motion_source"], output["predicted_future_motion"])
    assert output["z_target"].shape == (1, 2, 2)
    assert output["z_condition_active"] is False
    assert output["z_condition_step"] == 0
    assert model.z_condition_step.item() == 1
    assert "z_condition_step" in model.state_dict()
    assert actions.shape == (1, 2, 2)

    oracle_motion = torch.full((1, 4), 7.0)
    oracle_output = model.forward(
        **inputs,
        actions=torch.ones(1, 2, 2),
        z_motion_source=oracle_motion,
        noise=torch.zeros(1, 2, 2),
        z_noise=torch.zeros(1, 2, 2),
        time=torch.full((1,), 0.5),
        compute_motion_loss=False,
    )
    assert torch.equal(oracle_output["z_motion_source"], oracle_motion)
    assert oracle_output["z_condition_active"] is True
    assert oracle_output["z_condition_step"] == 1
    assert model.z_condition_step.item() == 2


def test_policy_combines_masked_flow_and_motion_losses():
    class _FlowModel(nn.Module):
        def forward(self, *args, **kwargs):
            del args
            assert kwargs["z_motion_source"] is None
            assert kwargs["compute_motion_loss"]
            assert kwargs["compute_flow"]
            return {
                "flow_losses": torch.tensor([[[1.0], [4.0], [9.0]]]),
                "z_flow_losses": torch.tensor([2.0]),
                "motion_losses": torch.tensor([4.0]),
                "predicted_future_motion": torch.tensor([[2.0]]),
                "z_target": torch.tensor([[3.0]]),
                "z_condition_active": True,
                "z_condition_step": 12,
            }

    class _MotionExtractor:
        @staticmethod
        def encode_pair(past_frames, future_frames):
            del past_frames, future_frames
            return torch.tensor([[0.0]]), torch.tensor([[1.0]])

    policy = SmolWPolicy.__new__(SmolWPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        action_feature=SimpleNamespace(shape=(1,)),
        motion_loss_weight=0.5,
        z_loss_weight=0.25,
        train_mode="jointly",
    )
    policy.model = _FlowModel()
    policy.motion_extractor = _MotionExtractor()
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: batch[ACTION]
    policy.prepare_motion_clips = lambda batch: (torch.empty(1), torch.empty(1))
    batch = {
        ACTION: torch.zeros(1, 3, 1),
        "action_is_pad": torch.tensor([[False, True, True]]),
        OBS_STATE: torch.zeros(1, 1),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }

    loss, metrics = policy.forward(batch)

    assert loss.item() == pytest.approx(3.5)
    assert metrics["flow_loss"] == pytest.approx(1.5)
    assert metrics["action_flow_loss"] == pytest.approx(1.0)
    assert metrics["z_flow_loss"] == pytest.approx(2.0)
    assert metrics["effective_z_loss_weight"] == pytest.approx(0.25)
    assert metrics["motion_loss"] == pytest.approx(4.0)
    assert metrics["z_condition_active"] == 1.0
    assert metrics["z_condition_step"] == 12.0


def test_motion_only_does_not_prepare_or_run_action_expert():
    class _MotionModel(nn.Module):
        def forward(self, *args, **kwargs):
            del args
            assert kwargs["actions"] is None
            assert kwargs["z_motion_source"] is None
            assert kwargs["compute_motion_loss"]
            assert not kwargs["compute_flow"]
            return {
                "motion_losses": torch.tensor([4.0]),
                "predicted_future_motion": torch.tensor([[2.0]]),
            }

    class _MotionExtractor:
        @staticmethod
        def encode_pair(past_frames, future_frames):
            del past_frames, future_frames
            return torch.tensor([[0.0]]), torch.tensor([[1.0]])

    policy = SmolWPolicy.__new__(SmolWPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        motion_loss_weight=0.5,
        z_loss_weight=1.0,
        train_mode="motion_only",
    )
    policy.model = _MotionModel()
    policy.motion_extractor = _MotionExtractor()
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: pytest.fail("motion_only must not prepare actions")
    policy.prepare_motion_clips = lambda batch: (torch.empty(1), torch.empty(1))
    batch = {
        OBS_STATE: torch.zeros(1, 1),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }

    loss, metrics = policy.forward(batch)

    assert loss.item() == pytest.approx(2.0)
    assert "flow_loss" not in metrics
    assert metrics["motion_loss"] == pytest.approx(4.0)


def test_action_only_uses_oracle_future_motion_without_motion_loss():
    class _ActionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_scale = nn.Parameter(torch.tensor(1.0))
            self.z_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, *args, **kwargs):
            del args
            assert kwargs["actions"] is not None
            assert torch.equal(kwargs["future_motion_target"], torch.tensor([[1.0]]))
            assert torch.equal(kwargs["z_motion_source"], torch.tensor([[1.0]]))
            assert not kwargs["compute_motion_loss"]
            assert kwargs["compute_flow"]
            return {
                "flow_losses": self.action_scale * torch.tensor([[[1.0], [4.0]]]),
                "z_flow_losses": self.z_scale * torch.tensor([0.5]),
                "predicted_future_motion": torch.tensor([[2.0]]),
                "z_motion_source": kwargs["z_motion_source"],
                "z_target": torch.tensor([[3.0]]),
                "z_condition_active": False,
                "z_condition_step": 3,
            }

    class _MotionExtractor:
        @staticmethod
        def encode_pair(past_frames, future_frames):
            del past_frames, future_frames
            return torch.tensor([[0.0]]), torch.tensor([[1.0]])

    policy = SmolWPolicy.__new__(SmolWPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        action_feature=SimpleNamespace(shape=(1,)),
        motion_loss_weight=1.0,
        z_loss_weight=1.0,
        train_mode="action_only",
    )
    policy.model = _ActionModel()
    policy.motion_extractor = _MotionExtractor()
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: batch[ACTION]
    policy.prepare_motion_clips = lambda batch: (torch.empty(1), torch.empty(1))
    batch = {
        ACTION: torch.zeros(1, 2, 1),
        OBS_STATE: torch.zeros(1, 1),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }

    loss, metrics = policy.forward(batch)

    assert loss.item() == pytest.approx(2.5)
    assert metrics["flow_loss"] == pytest.approx(2.5)
    assert metrics["action_flow_loss"] == pytest.approx(2.5)
    assert metrics["z_flow_loss"] == pytest.approx(0.5)
    assert metrics["weighted_z_flow_loss"] == pytest.approx(0.0)
    assert metrics["effective_z_loss_weight"] == pytest.approx(0.0)
    assert metrics["oracle_motion_rms"] == pytest.approx(1.0)
    assert metrics["predicted_motion_rms"] == pytest.approx(2.0)
    assert metrics["z_condition_active"] == 0.0
    assert metrics["z_condition_step"] == 3.0
    assert "motion_loss" not in metrics

    loss.backward()
    assert policy.model.action_scale.grad.item() == pytest.approx(2.5)
    assert policy.model.z_scale.grad.item() == pytest.approx(0.0)
