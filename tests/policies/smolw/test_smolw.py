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
    assert not config.train_expert_only
    assert not config.vidtwin_sample_posterior
    assert not hasattr(config, "vidtwin_repo_path")
    assert not hasattr(config, "vidtwin_config_path")
    assert not config.tensorboard_enable
    assert config.tensorboard_log_freq == 100
    for removed_field in (
        "train_mode",
        "training_stage",
        "motion_loss_weight",
        "motion_condition_hidden_dim",
        "motion_condition_scale",
        "detach_motion_condition",
        "z_condition_warmup_steps",
    ):
        assert not hasattr(config, removed_field)


def test_config_rejects_expert_only_training():
    with pytest.raises(ValueError, match="jointly trains"):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON,
            train_expert_only=True,
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


def test_config_can_be_saved_and_loaded(tmp_path):
    config = SmolWConfig(
        chunk_size=HORIZON,
        n_action_steps=HORIZON,
        motion_horizon=HORIZON,
        device="cpu",
    )
    config.save_pretrained(tmp_path)

    loaded_config = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded_config, SmolWConfig)
    assert loaded_config.observation_delta_indices == list(range(-15, HORIZON + 1))
    assert loaded_config.drop_n_last_frames == HORIZON
    assert not hasattr(loaded_config, "train_mode")


@pytest.mark.parametrize("z_loss_weight", [-1.0, 0.0])
def test_z_loss_weight_must_be_positive(z_loss_weight):
    with pytest.raises(ValueError, match="z_loss_weight"):
        SmolWConfig(
            chunk_size=HORIZON,
            n_action_steps=HORIZON,
            motion_horizon=HORIZON,
            z_loss_weight=z_loss_weight,
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


def test_motion_to_z_uses_fixed_per_slot_normalization():
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        motion_latent_dim=8,
        vidtwin_num_frames=2,
        motion_token_dim=4,
    )
    future_motion = torch.tensor([[1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0]])

    z_target = model.motion_to_z(future_motion)
    expected = torch.nn.functional.layer_norm(future_motion.reshape(1, 2, 4), (4,))

    assert z_target.shape == (1, 2, 4)
    assert torch.allclose(z_target, expected)
    assert torch.allclose(z_target.mean(dim=-1), torch.zeros(1, 2), atol=1e-6)


def test_joint_z_action_forward_and_sampling_keep_smolvla_action_shapes():
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        motion_latent_dim=4,
        motion_horizon=2,
        vidtwin_num_frames=2,
        motion_token_dim=2,
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
    model.z_token_in_proj = nn.Linear(2, 4)
    model.z_time_mlp_in = nn.Linear(8, 4)
    model.z_time_mlp_out = nn.Linear(4, 4)
    model.z_token_out_proj = nn.Linear(4, 2)
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
        future_motion=torch.zeros(1, 4),
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
    assert output["z_target"].shape == (1, 2, 2)
    assert actions.shape == (1, 2, 2)


def test_policy_always_jointly_trains_action_and_z_flow():
    class _JointFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_scale = nn.Parameter(torch.tensor(1.0))
            self.z_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, *args, **kwargs):
            assert torch.equal(args[-2], torch.zeros(1, 3, 1))
            assert torch.equal(args[-1], torch.tensor([[1.0]]))
            assert set(kwargs) == {"noise", "z_noise", "time"}
            return {
                "flow_losses": self.action_scale * torch.tensor([[[1.0], [4.0], [9.0]]]),
                "z_flow_losses": self.z_scale * torch.tensor([2.0]),
                "z_target": torch.tensor([[3.0]]),
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
        z_loss_weight=0.25,
    )
    policy.model = _JointFlowModel()
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

    assert loss.item() == pytest.approx(1.5)
    assert metrics["flow_loss"] == pytest.approx(1.5)
    assert metrics["action_flow_loss"] == pytest.approx(1.0)
    assert metrics["z_flow_loss"] == pytest.approx(2.0)
    assert metrics["weighted_z_flow_loss"] == pytest.approx(0.5)
    assert metrics["z_target_rms"] == pytest.approx(3.0)
    assert "motion_loss" not in metrics
    assert "predicted_motion_rms" not in metrics

    loss.backward()
    assert policy.model.action_scale.grad.item() == pytest.approx(1.0)
    assert policy.model.z_scale.grad.item() == pytest.approx(0.5)
