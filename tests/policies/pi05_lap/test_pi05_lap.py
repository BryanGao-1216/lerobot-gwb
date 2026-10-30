"""CPU checks for sampling, causal supervision and two-stage checkpoint transfer.

Use a tiny real PiGemma decoder and a local synthetic tokenizer: no CUDA, Hub,
training run, or production checkpoints are needed.
"""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("datasets")

from safetensors.torch import save_file
from tokenizers import Tokenizer, models
from transformers import GemmaConfig, PaliGemmaConfig, PreTrainedTokenizerFast

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05_lap import modeling_pi05_lap as lap
from lerobot.policies.pi05_lap.checkpoint import EMBEDDING_KEY, HEAD_KEY, load_student_state
from lerobot.policies.pi05_lap.configuration_pi05_lap import LAM_IMAGE_PAIR, PI05LAPConfig
from lerobot.policies.pi05_lap.dataset import PI05LAPDataset
from lerobot.policies.pi05_lap.lam.univla import load_lam_state
from lerobot.policies.pi05_lap.vocabulary import (
    LATENT_CODES,
    LATENT_START,
    TOKEN_MAP_FILE,
    LatentActionVocabulary,
    extend_vlm_vocabulary,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

CAMERA = "observation.images.image"
OTHER_CAMERA = "observation.images.image2"


def metadata():
    return SimpleNamespace(
        fps=20,
        features={
            CAMERA: {"dtype": "video", "shape": (24, 32, 3), "names": ["height", "width", "channels"]},
            OTHER_CAMERA: {"dtype": "image", "shape": (3, 24, 32), "names": ["channels", "height", "width"]},
            ACTION: {"dtype": "float32", "shape": (7,)},
        },
    )


def test_camera_selection_horizon_and_episode_clamping():
    cfg = PI05LAPConfig(training_stage="pre-training", chunk_size=12, n_action_steps=12, device="cpu")
    meta = metadata()
    delta = resolve_delta_timestamps(cfg, meta)
    assert cfg.lam_camera_key == CAMERA
    assert delta[CAMERA] == [0, 12 / 20]
    assert OTHER_CAMERA not in delta
    assert delta[ACTION] == [i / 20 for i in range(12)]
    assert replace(cfg, lam_camera_key="image2").resolve_lam_camera_key(meta.features) == OTHER_CAMERA
    with pytest.raises(ValueError, match="exactly one"):
        replace(cfg, lam_camera_key="image,image2")
    with pytest.raises(ValueError, match="not RGB"):
        replace(cfg, lam_camera_key="missing").resolve_lam_camera_key(meta.features)
    reader = DatasetReader.__new__(DatasetReader)
    reader._meta = SimpleNamespace(episodes=[{"dataset_from_index": 10, "dataset_to_index": 15}])
    reader.delta_indices = {CAMERA: [0, 12], ACTION: list(range(12))}
    indices, padding = reader._get_query_indices(13, 0)
    assert indices[CAMERA] == [13, 14]
    assert indices[ACTION] == [13] + [14] * 11
    assert padding[f"{CAMERA}_is_pad"].tolist() == [False, True]
    post_delta = resolve_delta_timestamps(replace(cfg, training_stage="post-training"), meta)
    assert CAMERA not in post_delta


def test_dataset_keeps_raw_teacher_and_only_current_student(monkeypatch):
    raw = torch.stack((torch.zeros(3, 24, 32), torch.ones(3, 24, 32)))
    actions = torch.randn(12, 7)

    def init(self, *args, **kwargs):
        assert kwargs["image_transforms"] is None
        self.meta = metadata()
        self.meta.camera_keys = [CAMERA, OTHER_CAMERA]
        self.meta.depth_keys = []

    monkeypatch.setattr(LeRobotDataset, "__init__", init)
    monkeypatch.setattr(
        LeRobotDataset,
        "__getitem__",
        lambda self, idx: {
            CAMERA: raw.clone(),
            OTHER_CAMERA: torch.zeros(3, 24, 32),
            ACTION: actions,
            f"{CAMERA}_is_pad": torch.tensor([False, True]),
        },
    )
    cfg = PI05LAPConfig(training_stage="pre-training", device="cpu")
    ds = PI05LAPDataset(policy_config=cfg, image_transforms=lambda x: x + 0.25)
    item = ds[0]
    assert item[CAMERA].shape == (3, 24, 32)
    assert torch.all(item[CAMERA] == 0.25)
    assert torch.all(item[OTHER_CAMERA] == 0.25)
    assert item[LAM_IMAGE_PAIR].shape == (2, 3, 224, 224)
    assert torch.all(item[LAM_IMAGE_PAIR][0] == 0)
    assert torch.all(item[LAM_IMAGE_PAIR][1] == 1)
    assert item[ACTION] is actions


def test_native_config_cli_conversion(tmp_path, monkeypatch):
    source = PI05Config(chunk_size=24, n_action_steps=12, max_state_dim=18, device="cpu")
    source._save_pretrained(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            "--policy.type=pi05_lap",
            f"--policy.path={tmp_path}",
            "--policy.training_stage=pre-training",
            "--policy.lam_model_path=/tmp/lam.ckpt",
            "--policy.chunk_size=12",
            "--policy.lam_camera_key=image2",
        ],
    )
    cfg = SimpleNamespace(resume=False)
    assert parser.filter_path_args("policy", sys.argv[1:]) == []
    TrainPipelineConfig._resolve_pretrained_from_cli(cfg)
    assert isinstance(cfg.policy, PI05LAPConfig)
    assert cfg.policy.training_stage == "pre-training"
    assert cfg.policy.chunk_size == 12
    assert cfg.policy.max_state_dim == 18
    assert cfg.policy.pretrained_path == tmp_path


@pytest.fixture
def tiny_policy(monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"<unk>": 0, **{f"w{i}": i for i in range(1, 32)}}, unk_token="<unk>")
        ),
        unk_token="<unk>",
        pad_token="<unk>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [LATENT_START, *LATENT_CODES]})
    vocabulary = LatentActionVocabulary(tokenizer)

    def small_config(variant):
        return lap.GemmaConfig(width=16, depth=2, mlp_dim=32, num_heads=8, num_kv_heads=1, head_dim=4)

    def init_wrapper(
        self, vlm_config, expert_config, *, freeze_vision_encoder=False, train_expert_only=False, **kwargs
    ):
        nn.Module.__init__(self)
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        text = {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "use_adarms": False,
            "adarms_cond_dim": None,
        }
        config = PaliGemmaConfig(
            text_config=text,
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
                "vision_use_head": False,
            },
            projection_dim=16,
            hidden_size=16,
            vocab_size=32,
        )
        self.paligemma = lap.PaliGemmaForConditionalGenerationWithPiGemma(config)
        expert = GemmaConfig(**{**text, "use_adarms": True, "adarms_cond_dim": 16})
        self.gemma_expert = lap.PiGemmaForCausalLM(expert)
        self.gemma_expert.model.embed_tokens = None
        self._set_requires_grad()

    monkeypatch.setattr(lap, "get_gemma_config", small_config)
    monkeypatch.setattr(lap.PaliGemmaWithExpertModel, "__init__", init_wrapper)

    def create(stage="pre-training", **kwargs):
        config = PI05LAPConfig(
            training_stage=stage,
            device="cpu",
            image_resolution=(8, 8),
            chunk_size=3,
            n_action_steps=3,
            max_action_dim=7,
            max_state_dim=7,
            **kwargs,
        )
        from lerobot.configs import FeatureType, PolicyFeature

        config.input_features = {CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 8, 8))}
        config.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (7,))}
        return lap.PI05LAPPolicy(config, _latent_vocabulary=vocabulary)

    return create


def batch():
    return {
        CAMERA: torch.rand(2, 3, 8, 8),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 3, 0], [2, 1, 0, 0]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, True, False], [True, True, False, False]]),
        ACTION: torch.randn(2, 3, 7),
        LAM_IMAGE_PAIR: torch.rand(2, 2, 3, 224, 224),
    }


def test_actual_decoder_has_no_future_label_leak(tiny_policy):
    policy = tiny_policy()
    policy.eval()
    data = batch()
    images, masks = policy._preprocess_images(data)
    suffix = torch.tensor([[32, 33, 34, 35], [32, 36, 37, 38]])
    args = (images, masks, data[OBS_LANGUAGE_TOKENS], data[OBS_LANGUAGE_ATTENTION_MASK])
    first = policy.model.latent_action_logits(*args, suffix)
    changed = suffix.clone()
    changed[:, -1] = 40
    second = policy.model.latent_action_logits(*args, changed)
    torch.testing.assert_close(first[:, :-1], second[:, :-1], rtol=0, atol=0)
    assert not torch.equal(first[:, -1], second[:, -1])


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_pretrain_gradients_and_native_action_loss(tiny_policy, gradient_checkpointing):
    policy = tiny_policy(freeze_vision_encoder=True, gradient_checkpointing=gradient_checkpointing)
    policy.train()
    teacher = SimpleNamespace(
        device=torch.device("cpu"), encode=lambda frames: torch.tensor([[0, 15, 2, 8]] * len(frames))
    )
    policy._lam_encoder = teacher
    data = batch()
    loss, _ = policy(data)
    loss.backward()
    vlm = policy.model.paligemma_with_expert.paligemma
    assert vlm.get_input_embeddings().weight.grad is not None
    assert vlm.lm_head.weight.grad is not None
    assert not vlm.model.vision_tower.training
    assert all(p.grad is None and not p.requires_grad for p in policy.model.action_in_proj.parameters())
    assert all(
        p.grad is None and not p.requires_grad
        for p in policy.model.paligemma_with_expert.gemma_expert.parameters()
    )
    predicted = policy.predict_latent_actions({k: v for k, v in data.items() if k != LAM_IMAGE_PAIR})
    assert predicted.shape == (2, 4)
    assert predicted.min() >= 0 and predicted.max() < 16
    post = tiny_policy("post-training")
    assert all(p.requires_grad for p in post.model.action_in_proj.parameters())
    del data[LAM_IMAGE_PAIR]
    rng = torch.get_rng_state()
    actual, _ = post(data)
    torch.set_rng_state(rng)
    expected, _ = PI05Policy.forward(post, data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert post._lam_encoder is None


def test_checkpoint_preserves_head_embedding_and_ids(tiny_policy, tmp_path):
    policy = tiny_policy()
    vlm = policy.model.paligemma_with_expert.paligemma
    with torch.no_grad():
        vlm.get_input_embeddings().weight.fill_(0.1)
        vlm.lm_head.weight.fill_(0.7)
    expected = {k: v.clone() for k, v in policy.state_dict().items()}
    policy.save_pretrained(tmp_path)
    config = replace(policy.config, training_stage="post-training", lam_model_path="/does/not/exist")
    restored = lap.PI05LAPPolicy.from_pretrained(tmp_path, config=config, local_files_only=True)
    assert restored.latent_vocabulary.metadata() == policy.latent_vocabulary.metadata()
    assert not any("lam_encoder" in key for key in expected)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    state = {k: v for k, v in expected.items() if k != "model.paligemma_with_expert.paligemma.lm_head.weight"}
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        lap.PI05LAPPolicy.from_pretrained(tmp_path, config=config, local_files_only=True)
    mapping = json.loads((tmp_path / TOKEN_MAP_FILE).read_text())
    mapping["code_ids"][0] += 1
    (tmp_path / TOKEN_MAP_FILE).write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="token IDs"):
        LatentActionVocabulary.from_pretrained(tmp_path, local_files_only=True)


def test_lam_rejects_wrong_checkpoint(tmp_path):
    checkpoint = tmp_path / "lam.ckpt"
    torch.save({"state_dict": {"vq_action.codebook.weight": torch.empty(16, 128)}}, checkpoint)
    with pytest.raises(ValueError, match="stage-2"):
        load_lam_state(checkpoint)
    state = {
        "vq_action.codebook.weight": torch.empty(16, 128),
        "action_latent_controllable": torch.empty(1, 1, 4, 768),
    }
    torch.save({"state_dict": {f"module.lam.{k}": v for k, v in state.items()}}, checkpoint)
    assert load_lam_state(checkpoint).keys() == state.keys()


def test_native_checkpoint_expands_only_new_rows(tiny_policy, tmp_path):
    policy = tiny_policy()
    vlm = policy.model.paligemma_with_expert.paligemma
    extend_vlm_vocabulary(vlm, 257169)
    new_rows = {
        EMBEDDING_KEY: vlm.get_input_embeddings().weight[257152:].detach().clone(),
        HEAD_KEY: vlm.lm_head.weight[257152:].detach().clone(),
    }
    state = {k: v.clone() for k, v in policy.state_dict().items()}
    state[EMBEDDING_KEY] = torch.full((257152, 16), 0.25)
    state[HEAD_KEY] = torch.full((257152, 16), 0.75)
    save_file(state, tmp_path / "model.safetensors")
    load_student_state(policy, tmp_path, source_type="pi05")
    restored = policy.state_dict()
    for key in (EMBEDDING_KEY, HEAD_KEY):
        torch.testing.assert_close(restored[key][:257152], state[key], rtol=0, atol=0)
        torch.testing.assert_close(restored[key][257152:], new_rows[key], rtol=0, atol=0)
