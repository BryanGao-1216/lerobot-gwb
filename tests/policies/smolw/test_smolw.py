from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.smolw.configuration_smolw import SmolWConfig
from lerobot.policies.smolw.modeling_smolw import SmolWFlowMatching, SmolWPolicy
from lerobot.policies.smolw.vidtwin_motion_encoder import VidTwinMotionExtractor
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
    assert config.future_motion_delta_indices == [0, 1, 2, 3]
    assert config.observation_delta_indices == [-6, -4, -2, 0, 1, 2, 3]
    assert config.past_motion_positions == [0, 1, 2, 3]
    assert config.future_motion_positions == [3, 4, 5, 6]
    assert config.current_observation_position == 3
    assert config.drop_n_last_frames == 3


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
        repo_path=tmp_path,
        config_path=tmp_path / "config.yaml",
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
        observation_delta_indices=[-6, -4, -2, 0, 1, 2, 3],
        current_observation_position=3,
        past_motion_positions=[0, 1, 2, 3],
        future_motion_positions=[3, 4, 5, 6],
        motion_horizon=4,
        memory_stride=2,
    )
    policy.motion_camera_key = CAMERA
    return policy


def test_current_observation_does_not_read_future_frames():
    policy = _temporal_policy()
    images = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1, 1, 1)
    states = torch.arange(14, dtype=torch.float32).reshape(1, 7, 2)
    batch = {
        CAMERA: images,
        f"{CAMERA}_is_pad": torch.tensor([[True, False, False, False, False, False, False]]),
        OBS_STATE: states,
    }

    current = policy._current_batch(batch)

    assert torch.equal(current[CAMERA], images[:, 3])
    assert torch.equal(current[OBS_STATE], states[:, 3])
    assert torch.equal(current[f"{CAMERA}_padding_mask"], torch.tensor([True]))


def test_motion_clips_use_strided_past_and_contiguous_future():
    policy = _temporal_policy()
    frames = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1, 1, 1)

    past, future = policy.prepare_motion_clips({CAMERA: frames})

    assert torch.equal(past.flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(future.flatten(), torch.tensor([3.0, 4.0, 5.0, 6.0]))


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
