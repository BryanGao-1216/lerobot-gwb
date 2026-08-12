import json
import math
import re

import pytest
import torch
from torch import nn

import lerobot.policies.smol_actionmem2.modeling_smol_actionmem as modeling_smol_actionmem2
from lerobot.datasets.rlds_dataset import resolve_actionmem_token_metadata
from lerobot.policies.actionmem.action_vqvae import ActionVQVAEQ0Encoder
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.smol_actionmem2.configuration_smol_actionmem import SmolActionMem2Config
from lerobot.policies.smol_actionmem2.modeling_smol_actionmem import (
    SmolActionMem2FlowMatching,
    SmolActionMem2Policy,
)
from lerobot.policies.smol_actionmem2.processor_smol_actionmem import (
    SmolActionMem2ActionCodeProcessorStep,
)
from lerobot.policies.smol_actionmem2.tokenization_smol_actionmem import SmolActionMem2TokenMap
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_TOKENS,
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
                # Legacy vocabulary fields are deliberately ignored.
                "action_tokens": {
                    "anchor_token_id": 49276,
                    "token_id_min": 49021,
                    "token_id_max": 49276,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _transition(q0=None, batch_size=2):
    data = {"task": ["pick"] * batch_size}
    if q0 is not None:
        data[ACTION_TOKEN] = q0
    return create_transition(
        observation={OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 4, dtype=torch.long)},
        complementary_data=data,
    )


def test_policy_is_independently_registered():
    config = make_policy_config("smol_actionmem2")

    assert isinstance(config, SmolActionMem2Config)
    assert get_policy_class("smol_actionmem2") is SmolActionMem2Policy


def test_config_rejects_non_positive_soft_target_temperature():
    with pytest.raises(ValueError, match="action_token_soft_target_temperature"):
        SmolActionMem2Config(action_token_soft_target_temperature=0)


def test_q0_encoder_exposes_the_same_latent_distances_used_for_assignment():
    class _FlatEncoder(nn.Module):
        horizon = 2
        action_dim = 1
        in_channels = 1

        def forward(self, actions):
            return actions.flatten(start_dim=1)

    encoder = ActionVQVAEQ0Encoder(
        encoder=_FlatEncoder(),
        q0_codebook=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        time_emb=None,
        xyz_emb=None,
        euler_emb=None,
        gripper_emb=None,
        action_mean=torch.zeros(1),
        action_std=torch.ones(1),
        normalize_actions=False,
        use_action_type_pe=False,
    )
    actions = torch.tensor([[[0.75], [1.25]]])

    distances = encoder.compute_code_distances(actions)

    assert torch.allclose(distances, torch.tensor([[2.125, 0.125, 2.125]]))
    assert torch.equal(encoder(actions), distances.argmin(dim=-1))


def test_token_map_allocates_a_local_action_context(tmp_path):
    token_map = SmolActionMem2TokenMap.from_json(_write_token_map(tmp_path))

    assert token_map.action_class_min == 0
    assert token_map.action_class_max == 255
    assert token_map.memory_start_id == 256
    assert token_map.memory_end_id == 257
    assert token_map.action_query_id == 258
    assert token_map.padding_id == 259
    assert token_map.context_vocab_size == 260


def test_rlds_resolves_smol_actionmem2_metadata(tmp_path):
    config = type(
        "Config",
        (),
        {
            "type": "smol_actionmem2",
            "pretrained_path": str(tmp_path),
            "action_token_map_path": None,
        },
    )()
    _write_token_map(tmp_path)

    metadata = resolve_actionmem_token_metadata(config)

    assert metadata.codebook_size == 256
    assert metadata.action_horizon == 16


def test_processor_emits_local_classes_instead_of_language_token_ids(tmp_path):
    step = SmolActionMem2ActionCodeProcessorStep(str(_write_token_map(tmp_path)))

    output = step(_transition(torch.tensor([[0], [255]])))[TransitionKey.COMPLEMENTARY_DATA]
    inference = step(_transition())[TransitionKey.COMPLEMENTARY_DATA]

    assert torch.equal(
        output[ACTION_TOKENS],
        torch.tensor([[256, 257, 258, 0], [256, 257, 258, 255]]),
    )
    assert output[ACTION_TOKEN_MASK].all()
    assert torch.equal(
        inference[ACTION_TOKENS],
        torch.tensor([[256, 257, 258, 259], [256, 257, 258, 259]]),
    )
    assert not inference[ACTION_TOKEN_MASK][:, -1].any()


class _DummyVLM:
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        return tokens.float().unsqueeze(-1).expand(-1, -1, 4)


def test_training_flow_source_is_standard_gaussian_noise():
    model = SmolActionMem2FlowMatching.__new__(SmolActionMem2FlowMatching)
    nn.Module.__init__(model)
    actions = torch.zeros(2, 3, 4)
    expected_noise = torch.arange(actions.numel(), dtype=actions.dtype).reshape_as(actions)
    model.sample_noise = lambda shape, device: expected_noise

    source = model._make_training_flow_source(actions)

    assert source is expected_noise


def test_prefix_uses_independent_action_embedding_and_keeps_state_before_query():
    model = SmolActionMem2FlowMatching.__new__(SmolActionMem2FlowMatching)
    nn.Module.__init__(model)
    model.vlm_with_expert = _DummyVLM()
    model.action_code_map = type(
        "Map",
        (),
        {"action_query_id": 258, "action_class_min": 0, "action_class_max": 255},
    )()
    model.action_code_embedding = nn.Embedding(260, 4)
    with torch.no_grad():
        model.action_code_embedding.weight.copy_(
            torch.arange(260, dtype=torch.float32)[:, None].expand(-1, 4)
        )
    model.add_image_special_tokens = False
    model.prefix_length = -1
    model.state_proj = nn.Linear(2, 4, bias=False)
    model.state_proj.weight.data.copy_(torch.eye(4, 2))

    embeddings, _, _, query_positions = model.embed_prefix(
        images=[torch.ones(1, 2, 4)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[10, 11, 2]]),
        lang_masks=torch.tensor([[True, True, False]]),
        action_tokens=torch.tensor([[256, 257, 258, 3]]),
        action_token_masks=torch.ones(1, 4, dtype=torch.bool),
        state=torch.tensor([[1.0, 2.0]]),
    )

    assert torch.equal(embeddings[0, 5:7, 0] / math.sqrt(4), torch.tensor([256.0, 257.0]))
    assert torch.equal(embeddings[0, 7], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert torch.equal(embeddings[0, 8:10, 0] / math.sqrt(4), torch.tensor([258.0, 3.0]))
    assert query_positions.item() == 8


def test_action_objective_is_exactly_256_way():
    model = SmolActionMem2FlowMatching.__new__(SmolActionMem2FlowMatching)
    nn.Module.__init__(model)
    model.config = type("Config", (), {"action_token_soft_target_temperature": 1.0})()
    model.action_classifier = nn.Linear(4, 256)
    with torch.no_grad():
        model.action_classifier.weight.zero_()
        model.action_classifier.bias.zero_()
        model.action_classifier.weight[3, 0] = 5.0

    output = model._compute_action_token_objective(
        prefix_out=torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        query_positions=torch.tensor([1]),
        action_tokens=torch.tensor([[256, 257, 258, 3]]),
        action_token_masks=torch.ones(1, 4, dtype=torch.bool),
        q0_distances=torch.nn.functional.one_hot(torch.tensor([3]), 256).logical_not().float(),
    )

    assert model.action_classifier.out_features == 256
    assert output["action_token_accuracy"].item() == 1.0


def test_action_objective_ignores_out_of_range_local_padding_class():
    model = SmolActionMem2FlowMatching.__new__(SmolActionMem2FlowMatching)
    nn.Module.__init__(model)
    model.action_classifier = nn.Linear(4, 256)

    output = model._compute_action_token_objective(
        prefix_out=torch.zeros(1, 2, 4),
        query_positions=torch.tensor([1]),
        action_tokens=torch.tensor([[256, 257, 258, 259]]),
        action_token_masks=torch.tensor([[True, True, True, False]]),
        q0_distances=None,
    )

    assert output["action_token_kl_loss"].item() == 0
    assert output["action_token_accuracy"].item() == 0


def test_action_objective_matches_latent_distance_soft_target_kl():
    model = SmolActionMem2FlowMatching.__new__(SmolActionMem2FlowMatching)
    nn.Module.__init__(model)
    model.config = type("Config", (), {"action_token_soft_target_temperature": 2.0})()
    model.action_classifier = nn.Linear(1, 3)
    with torch.no_grad():
        model.action_classifier.weight.zero_()
        model.action_classifier.bias.copy_(torch.tensor([1.0, 0.0, -1.0]))

    distances = torch.tensor([[0.0, 2.0, 4.0]])
    soft_target = torch.softmax(-distances / 2.0, dim=-1)
    expected_kl = torch.nn.functional.kl_div(
        torch.log_softmax(model.action_classifier.bias, dim=-1).unsqueeze(0),
        soft_target,
        reduction="batchmean",
    )
    output = model._compute_action_token_objective(
        prefix_out=torch.zeros(1, 1, 1),
        query_positions=torch.tensor([0]),
        action_tokens=torch.tensor([[3, 4, 5, 0]]),
        action_token_masks=torch.ones(1, 4, dtype=torch.bool),
        q0_distances=distances,
    )

    assert torch.allclose(output["action_token_kl_loss"], expected_kl)
    assert 0 < output["action_token_soft_target_entropy"].item() < math.log(3)
    assert 0 < output["action_token_soft_target_peak_probability"].item() < 1


class _InferenceVLM(nn.Module):
    def embed_language_tokens(self, tokens):
        return tokens.float().unsqueeze(-1).expand(-1, -1, 4)

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        return [inputs_embeds[0], None], "cache"


def test_inference_uses_classifier_class_as_condition_with_gaussian_noise(monkeypatch):
    model = SmolActionMem2FlowMatching.__new__(SmolActionMem2FlowMatching)
    nn.Module.__init__(model)
    model.config = type(
        "Config",
        (),
        {
            "chunk_size": 16,
            "max_action_dim": 7,
            "num_inference_steps": 1,
            "use_cache": False,
            "rtc_config": None,
        },
    )()
    model.rtc_processor = None
    model.vlm_with_expert = _InferenceVLM()
    model.action_code_map = type(
        "Map",
        (),
        {"action_query_id": 258, "action_class_min": 0, "action_class_max": 255},
    )()
    model.action_code_embedding = nn.Embedding(260, 4)
    model.action_classifier = nn.Linear(4, 256)
    with torch.no_grad():
        model.action_classifier.weight.zero_()
        model.action_classifier.bias.zero_()
        model.action_classifier.bias[7] = 10.0
    model.state_proj = nn.Linear(2, 4)
    model.add_image_special_tokens = False
    model.prefix_length = -1
    embedded_action_sequences = []
    model.action_code_embedding.register_forward_pre_hook(
        lambda module, args: embedded_action_sequences.append(args[0].clone())
    )
    expected_noise = torch.ones(1, 16, 7)
    model.sample_noise = lambda shape, device: expected_noise
    monkeypatch.setattr(
        modeling_smol_actionmem2,
        "euler_integrate",
        lambda denoise_fn, noise, num_steps, **kwargs: noise,
    )

    output = model.sample_actions(
        images=[],
        img_masks=[],
        lang_tokens=torch.tensor([[10, 11]]),
        lang_masks=torch.ones(1, 2, dtype=torch.bool),
        action_tokens=torch.tensor([[256, 257, 258, 259]]),
        action_token_masks=torch.tensor([[True, True, True, False]]),
        state=torch.zeros(1, 2),
    )

    assert output is expected_noise
    assert torch.equal(embedded_action_sequences[0], torch.tensor([[256, 257, 258]]))
    assert torch.equal(embedded_action_sequences[1], torch.tensor([[256, 257, 258, 7]]))


class _StageModel(SmolActionMem2FlowMatching):
    def __init__(self, stage):
        nn.Module.__init__(self)
        self.config = type("Config", (), {"training_stage": stage, "freeze_vision_encoder": False})()
        self.vlm_with_expert = nn.Module()
        self.vlm_with_expert.vlm = nn.Module()
        self.vlm_with_expert.vlm.backbone = nn.Linear(2, 2)
        self.vlm_with_expert.vlm.lm_head = nn.Linear(2, 2)
        self.vlm_with_expert.lm_expert = nn.Linear(2, 2)
        self.state_proj = nn.Linear(2, 2)
        self.action_code_embedding = nn.Embedding(260, 2)
        self.action_classifier = nn.Linear(2, 256)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.action_time_mlp_in = nn.Linear(2, 2)
        self.action_time_mlp_out = nn.Linear(2, 2)
        self._training_stage_configured = False


@pytest.mark.parametrize(
    ("stage", "classifier_trainable", "expert_trainable"),
    [
        ("vlm_only", True, False),
        ("action_expert_only", False, True),
        ("joint", True, True),
    ],
)
def test_training_stage_handles_new_action_modules(stage, classifier_trainable, expert_trainable):
    model = _StageModel(stage)
    model.configure_training_stage()

    assert all(
        parameter.requires_grad is classifier_trainable for parameter in model.action_classifier.parameters()
    )
    assert all(
        parameter.requires_grad is expert_trainable for parameter in model.action_out_proj.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.action_code_embedding.parameters())
    assert not any(parameter.requires_grad for parameter in model.vlm_with_expert.vlm.lm_head.parameters())


def test_default_peft_targets_do_not_train_language_vocabulary():
    policy = SmolActionMem2Policy.__new__(SmolActionMem2Policy)
    defaults = policy._get_default_peft_targets()
    pattern = defaults["target_modules"]

    assert re.fullmatch(
        pattern,
        "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj",
    )
    assert not re.fullmatch(pattern, "model.vlm_with_expert.vlm.lm_head")
    assert not re.fullmatch(pattern, "model.vlm_with_expert.vlm.model.text_model.embed_tokens")
    assert defaults["modules_to_save"] == ["action_code_embedding", "action_classifier"]
