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


def test_policy_is_independently_registered():
    config = make_policy_config(
        "smolw",
        chunk_size=4,
        n_action_steps=4,
        motion_horizon=4,
        device="cpu",
    )

    assert isinstance(config, SmolWConfig)
    assert get_policy_class("smolw") is SmolWPolicy


def test_config_builds_past_and_future_lerobot_timestamps():
    config = SmolWConfig(
        chunk_size=4,
        n_action_steps=4,
        motion_horizon=4,
        memory_stride=2,
        device="cpu",
    )

    assert config.past_motion_delta_indices == [-6, -4, -2, 0]
    assert config.future_motion_delta_indices == [1, 2, 3, 4]
    assert config.observation_delta_indices == [-6, -4, -2, 0, 1, 2, 3, 4]
    assert config.past_motion_positions == [0, 1, 2, 3]
    assert config.future_motion_positions == [4, 5, 6, 7]
    assert config.current_observation_position == 3
    assert config.drop_n_last_frames == 4
    assert config.training_stage == "world_model"
    assert not config.vidtwin_sample_posterior
    assert not hasattr(config, "vidtwin_repo_path")
    assert not hasattr(config, "vidtwin_config_path")
    assert not config.tensorboard_enable
    assert config.tensorboard_log_freq == 100


def test_action_expert_stage_requests_only_history_and_keeps_episode_tail():
    config = SmolWConfig(
        chunk_size=4,
        n_action_steps=4,
        motion_horizon=4,
        memory_stride=2,
        training_stage="action_expert_only",
        device="cpu",
    )

    assert config.observation_delta_indices == [-6, -4, -2, 0]
    assert config.past_motion_positions == [0, 1, 2, 3]
    assert config.future_motion_positions == []
    assert config.drop_n_last_frames == 0


def test_config_rejects_unknown_training_stage():
    with pytest.raises(ValueError, match="training_stage"):
        SmolWConfig(
            chunk_size=4,
            n_action_steps=4,
            motion_horizon=4,
            training_stage="unknown",
            device="cpu",
        )


def test_config_enforces_fixed_vidtwin_frame_count():
    with pytest.raises(ValueError, match="vidtwin_num_frames=16"):
        SmolWConfig(
            chunk_size=4,
            n_action_steps=4,
            motion_horizon=4,
            vidtwin_num_frames=8,
            device="cpu",
        )


def test_stage_one_config_can_be_loaded_with_stage_two_overrides(tmp_path):
    stage_one = SmolWConfig(
        chunk_size=4,
        n_action_steps=4,
        motion_horizon=4,
        training_stage="world_model",
        device="cpu",
    )
    stage_one.save_pretrained(tmp_path)

    stage_two = PreTrainedConfig.from_pretrained(
        tmp_path,
        cli_overrides=[
            "--training_stage=action_expert_only",
            "--drop_n_last_frames=0",
        ],
    )

    assert isinstance(stage_two, SmolWConfig)
    assert stage_two.training_stage == "action_expert_only"
    assert stage_two.observation_delta_indices == [-3, -2, -1, 0]
    assert stage_two.drop_n_last_frames == 0


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
            chunk_size=4,
            n_action_steps=4,
            motion_horizon=4,
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
            chunk_size=4,
            n_action_steps=4,
            motion_horizon=3,
            device="cpu",
        )


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


def _stage_model(stage: str) -> SmolWFlowMatching:
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(training_stage=stage)
    model.vlm_with_expert = _StageVLMWithExpert()
    model.state_proj = nn.Linear(2, 2)
    model.mt_query_embedding = nn.Embedding(1, 2)
    model.past_motion_projector = nn.Linear(2, 2)
    model.future_motion_head = nn.Linear(2, 2)
    model.future_motion_condition_proj = nn.Linear(2, 2)
    model.future_visual_queries = nn.Embedding(1, 2)
    model.future_motion_visual_proj = nn.Linear(2, 2)
    model.future_visual_decoder = nn.Linear(2, 2)
    model.future_visual_out_proj = nn.Linear(2, 2)
    model.action_in_proj = nn.Linear(2, 2)
    model.action_out_proj = nn.Linear(2, 2)
    model.action_time_mlp_in = nn.Linear(2, 2)
    model.action_time_mlp_out = nn.Linear(2, 2)
    model._training_stage_configured = False
    return model


def test_training_stage_freezes_the_other_branch_and_visual_teacher():
    world_model = _stage_model("world_model")
    # Simulate a tensor SmolVLA already froze to avoid DDP unused parameters.
    world_model.vlm_with_expert.vlm.model.text_model.bias.requires_grad_(False)
    world_model.configure_training_stage()

    assert world_model.vlm_with_expert.vlm.model.text_model.weight.requires_grad
    assert not world_model.vlm_with_expert.vlm.model.text_model.bias.requires_grad
    assert not world_model.vlm_with_expert.vlm.model.vision_model.weight.requires_grad
    assert not world_model.vlm_with_expert.vlm.model.connector.weight.requires_grad
    assert world_model.future_motion_head.weight.requires_grad
    assert world_model.future_visual_out_proj.weight.requires_grad
    assert not world_model.vlm_with_expert.lm_expert.weight.requires_grad
    assert not world_model.future_motion_condition_proj.weight.requires_grad

    action_model = _stage_model("action_expert_only")
    action_model.configure_training_stage()

    assert not action_model.vlm_with_expert.vlm.model.text_model.weight.requires_grad
    assert not action_model.future_motion_head.weight.requires_grad
    assert not action_model.future_visual_out_proj.weight.requires_grad
    assert action_model.vlm_with_expert.lm_expert.weight.requires_grad
    assert action_model.action_in_proj.weight.requires_grad
    assert action_model.future_motion_condition_proj.weight.requires_grad

    action_model.train()
    assert not action_model.vlm_with_expert.vlm.training
    assert action_model.vlm_with_expert.lm_expert.training


def test_future_visual_loss_backpropagates_through_predicted_motion():
    class _PassThroughDecoder(nn.Module):
        @staticmethod
        def forward(tgt, memory):
            del memory
            return tgt

    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(future_visual_cosine_weight=0.1)
    model.future_visual_num_tokens = 2
    model.future_visual_queries = nn.Embedding(2, 4)
    nn.init.zeros_(model.future_visual_queries.weight)
    model.future_motion_visual_proj = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4, bias=False))
    nn.init.eye_(model.future_motion_visual_proj[1].weight)
    model.future_visual_decoder = _PassThroughDecoder()
    model.future_visual_out_proj = nn.Linear(4, 4, bias=False)
    nn.init.eye_(model.future_visual_out_proj.weight)
    model.encode_visual_teacher = lambda image: image

    current_tokens = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    future_tokens = current_tokens + 0.5
    predicted_motion = torch.tensor([[0.0, 1.0, 2.0, 4.0]], requires_grad=True)

    losses = model.compute_future_visual_losses(
        current_tokens,
        future_tokens,
        predicted_motion,
    )
    losses["future_visual_losses"].mean().backward()

    assert losses["future_visual_losses"].shape == (1,)
    assert losses["copy_current_visual_losses"].item() > 0
    assert predicted_motion.grad is not None
    assert torch.any(predicted_motion.grad != 0)


def test_visual_token_count_matches_smolvlm_connector_grid():
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(resize_imgs_with_padding=(512, 512))
    model.vlm_with_expert = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=16),
            scale_factor=4,
        )
    )

    assert model._infer_visual_token_count() == 64


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


def test_action_expert_cannot_attend_mt_query_directly():
    prefix_masks = torch.tensor([[True, True, False, True]])
    suffix_masks = torch.tensor([[True, True]])
    suffix_attention = torch.ones(1, 2, dtype=torch.bool)

    attention, position_ids = SmolWFlowMatching.make_action_attention(
        prefix_masks,
        suffix_masks,
        suffix_attention,
    )

    assert attention.shape == (1, 2, 6)
    assert not attention[:, :, 2].any()
    assert not attention[:, :, 3].any()
    assert attention[:, :, :2].all()
    assert torch.equal(position_ids, torch.tensor([[2, 3]]))


def test_motion_first_forward_and_sampling_keep_smolvla_action_shapes():
    model = SmolWFlowMatching.__new__(SmolWFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        motion_latent_dim=3,
        motion_condition_scale=1.0,
        detach_motion_condition=False,
        chunk_size=2,
        max_action_dim=2,
        min_period=4e-3,
        max_period=4.0,
        num_steps=2,
        rtc_config=None,
        future_visual_loss_weight=0.0,
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
    model.past_motion_projector = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 4))
    model.future_motion_head = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 3))
    model.future_motion_condition_proj = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, 4))
    model.rtc_processor = None

    inputs = {
        "images": [torch.ones(1, 2, 4)],
        "img_masks": [torch.tensor([True])],
        "lang_tokens": torch.tensor([[2, 3]]),
        "lang_masks": torch.tensor([[True, True]]),
        "state": torch.ones(1, 2),
        "past_motion": torch.ones(1, 3),
    }
    output = model.forward(
        **inputs,
        actions=torch.ones(1, 2, 2),
        future_motion_target=torch.zeros(1, 3),
        noise=torch.zeros(1, 2, 2),
        time=torch.full((1,), 0.5),
    )
    actions = model.sample_actions(**inputs, noise=torch.zeros(1, 2, 2))

    assert output["flow_losses"].shape == (1, 2, 2)
    assert output["motion_losses"].shape == (1,)
    assert output["predicted_future_motion"].shape == (1, 3)
    assert actions.shape == (1, 2, 2)


def test_policy_combines_masked_flow_and_motion_losses():
    class _FlowModel(nn.Module):
        def forward(self, *args, **kwargs):
            del args, kwargs
            return {
                "flow_losses": torch.tensor([[[1.0], [4.0], [9.0]]]),
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
        action_feature=SimpleNamespace(shape=(1,)),
        motion_loss_weight=0.5,
        future_visual_loss_weight=0.0,
        training_stage="joint",
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

    assert loss.item() == pytest.approx(3.0)
    assert metrics["flow_loss"] == pytest.approx(1.0)
    assert metrics["motion_loss"] == pytest.approx(4.0)


def test_world_model_stage_does_not_prepare_or_run_action_expert():
    class _WorldModel(nn.Module):
        def forward(self, *args, **kwargs):
            del args
            assert kwargs["actions"] is None
            assert kwargs["compute_world_model"]
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
        future_visual_loss_weight=0.0,
        training_stage="world_model",
    )
    policy.model = _WorldModel()
    policy.motion_extractor = _MotionExtractor()
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: pytest.fail("stage one must not prepare actions")
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


def test_action_expert_stage_does_not_read_future_supervision():
    class _ActionModel(nn.Module):
        def forward(self, *args, **kwargs):
            del args
            assert kwargs["actions"] is not None
            assert kwargs["future_motion_target"] is None
            assert kwargs["current_visual_image"] is None
            assert kwargs["future_visual_image"] is None
            assert not kwargs["compute_world_model"]
            assert kwargs["compute_flow"]
            return {
                "flow_losses": torch.tensor([[[1.0], [4.0]]]),
                "predicted_future_motion": torch.tensor([[2.0]]),
            }

    class _MotionExtractor:
        @staticmethod
        def encode(past_frames):
            del past_frames
            return torch.tensor([[0.0]])

    policy = SmolWPolicy.__new__(SmolWPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        action_feature=SimpleNamespace(shape=(1,)),
        motion_loss_weight=1.0,
        future_visual_loss_weight=1.0,
        training_stage="action_expert_only",
    )
    policy.model = _ActionModel()
    policy.motion_extractor = _MotionExtractor()
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: batch[ACTION]
    policy.prepare_past_motion_clip = lambda batch: torch.empty(1)
    policy.prepare_motion_clips = lambda batch: pytest.fail("stage two must not request future frames")
    policy.prepare_future_visual_pair = lambda batch: pytest.fail(
        "stage two must not request the future visual target"
    )
    batch = {
        ACTION: torch.zeros(1, 2, 1),
        OBS_STATE: torch.zeros(1, 1),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }

    loss, metrics = policy.forward(batch)

    assert loss.item() == pytest.approx(2.5)
    assert metrics["flow_loss"] == pytest.approx(2.5)
    assert "motion_loss" not in metrics
