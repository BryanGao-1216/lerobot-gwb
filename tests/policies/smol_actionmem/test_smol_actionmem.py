import math
import re

import pytest
import torch
from torch import nn

import lerobot.policies.smol_actionmem.modeling_smol_actionmem as modeling_smol_actionmem
from lerobot.policies.action_code import ActionCodeLayout
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.smol_actionmem.configuration_smol_actionmem import SmolActionMemConfig
from lerobot.policies.smol_actionmem.modeling_smol_actionmem import (
    SmolActionMemFlowMatching,
    SmolActionMemPolicy,
)
from lerobot.policies.smol_actionmem.processor_smol_actionmem import (
    SmolActionMemActionCodeProcessorStep,
)
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN,
    ACTION_TOKEN_MASK,
    ACTION_TOKENIZER_INPUT,
    ACTION_TOKENS,
    OBS_LANGUAGE_TOKENS,
)


def _transition(action_codes=None, batch_size=2):
    data = {"task": ["pick"] * batch_size}
    if action_codes is not None:
        data[ACTION_TOKEN] = action_codes
    return create_transition(
        observation={OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 4, dtype=torch.long)},
        complementary_data=data,
    )


def test_policy_is_independently_registered():
    config = make_policy_config("smol_actionmem")

    assert isinstance(config, SmolActionMemConfig)
    assert get_policy_class("smol_actionmem") is SmolActionMemPolicy


def test_config_rejects_non_positive_soft_target_temperature():
    with pytest.raises(ValueError, match="action_token_soft_target_temperature"):
        SmolActionMemConfig(action_token_soft_target_temperature=0)


def test_processor_emits_local_classes_instead_of_language_token_ids():
    step = SmolActionMemActionCodeProcessorStep(codebook_size=256)

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


def test_flow_target_uses_preprocessor_action_not_tokenizer_input():
    policy = SmolActionMemPolicy.__new__(SmolActionMemPolicy)
    nn.Module.__init__(policy)
    policy.config = type(
        "Config",
        (),
        {
            "max_action_dim": 32,
        },
    )()
    processed_actions = torch.linspace(-2, 2, 2 * 16 * 7).reshape(2, 16, 7)
    tokenizer_actions = torch.linspace(-1, 1, 2 * 16 * 7).reshape(2, 16, 7)
    batch = {
        "action": processed_actions,
        ACTION_TOKENIZER_INPUT: tokenizer_actions,
    }

    target = policy.prepare_action(batch)

    assert target.shape == (2, 16, 32)
    assert torch.equal(target[..., :7], processed_actions)
    assert not torch.equal(target[..., :7], tokenizer_actions)
    assert torch.count_nonzero(target[..., 7:]) == 0


def test_prefix_uses_independent_action_embedding_and_keeps_state_before_query():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    model.vlm_with_expert = _DummyVLM()
    model.action_query_id = 258
    model.action_code_embedding = nn.Embedding(260, 4)
    with torch.no_grad():
        model.action_code_embedding.weight.copy_(
            torch.arange(260, dtype=torch.float32)[:, None].expand(-1, 4)
        )
    model.add_image_special_tokens = False
    model.prefix_length = 12
    model.state_proj = nn.Linear(2, 4, bias=False)
    model.state_proj.weight.data.copy_(torch.eye(4, 2))

    embeddings, padding_masks, _ = model.embed_prefix(
        images=[torch.ones(1, 2, 4)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[10, 11, 2]]),
        lang_masks=torch.tensor([[True, True, False]]),
        action_tokens=torch.tensor([[256, 257, 258]]),
        action_token_masks=torch.ones(1, 3, dtype=torch.bool),
        state=torch.tensor([[1.0, 2.0]]),
    )

    assert torch.equal(embeddings[0, 5:7, 0] / math.sqrt(4), torch.tensor([256.0, 257.0]))
    assert torch.equal(embeddings[0, 7], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert embeddings.shape[1] == 12
    assert not padding_masks[0, 8:11].any()
    assert padding_masks[0, -1]
    assert embeddings[0, -1, 0].item() / math.sqrt(4) == 258.0


def test_action_objective_is_exactly_256_way():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    model.config = type("Config", (), {"action_token_soft_target_temperature": 1.0})()
    model.action_classifier = nn.Linear(4, 256)
    with torch.no_grad():
        model.action_classifier.weight.zero_()
        model.action_classifier.bias.zero_()
        model.action_classifier.weight[3, 0] = 5.0

    logits = model._compute_action_logits(
        prefix_out=torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    )
    output = model._compute_action_token_objective(
        logits=logits,
        action_tokens=torch.tensor([[256, 257, 258, 3]]),
        action_token_masks=torch.ones(1, 4, dtype=torch.bool),
        action_code_distances=torch.nn.functional.one_hot(torch.tensor([3]), 256).logical_not().float(),
    )

    assert model.action_classifier.out_features == 256
    assert output["action_token_accuracy"].item() == 1.0
    assert output["action_token_target_rank"].item() == 1.0


def test_action_objective_ignores_out_of_range_local_padding_class():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    model.config = type("Config", (), {"action_token_soft_target_temperature": 1.0})()
    model.action_classifier = nn.Linear(4, 256)

    output = model._compute_action_token_objective(
        logits=torch.zeros(1, 256),
        action_tokens=torch.tensor([[256, 257, 258, 259]]),
        action_token_masks=torch.tensor([[True, True, True, False]]),
        action_code_distances=None,
    )

    assert output["action_token_kl_loss"].item() == 0
    assert output["action_token_accuracy"].item() == 0
    assert output["action_token_target_rank"].item() == 0


def test_action_objective_averages_target_logit_rank_over_valid_batch_items():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    model.config = type("Config", (), {"action_token_soft_target_temperature": 1.0})()
    model.action_classifier = nn.Linear(1, 3)

    output = model._compute_action_token_objective(
        logits=torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]]),
        action_tokens=torch.tensor([[3, 4, 5, 0], [3, 4, 5, 2], [3, 4, 5, 1]]),
        action_token_masks=torch.tensor(
            [[True, True, True, True], [True, True, True, True], [True, True, True, False]]
        ),
        action_code_distances=torch.zeros(3, 3),
    )

    # The valid targets rank first and third respectively; the masked sample is ignored.
    assert output["action_token_target_rank"].item() == 2.0


def test_action_objective_matches_latent_distance_soft_target_kl():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
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
        logits=model.action_classifier(torch.zeros(1, 1)),
        action_tokens=torch.tensor([[3, 4, 5, 0]]),
        action_token_masks=torch.ones(1, 4, dtype=torch.bool),
        action_code_distances=distances,
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


def test_inference_uses_complete_logits_condition_without_argmax_token(monkeypatch):
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
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
    model.action_code_layout = ActionCodeLayout()
    model.action_codebook_size = 256
    model.action_query_id = 258
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
    received_logits = []
    model._denoise_step_with_action_condition = lambda **kwargs: (
        received_logits.append(kwargs["action_logits"].detach().clone()) or torch.zeros_like(kwargs["x_t"])
    )
    monkeypatch.setattr(
        modeling_smol_actionmem,
        "euler_integrate",
        lambda denoise_fn, noise, num_steps, **kwargs: (
            denoise_fn(noise, torch.ones(noise.shape[0], device=noise.device)),
            noise,
        )[1],
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
    assert len(embedded_action_sequences) == 1
    assert len(received_logits) == 1
    assert received_logits[0].shape == (1, 256)
    assert received_logits[0][0, 7] == 10


def test_continuous_condition_receives_flow_gradients_but_blocks_logit_gradients():
    model = SmolActionMemFlowMatching.__new__(SmolActionMemFlowMatching)
    nn.Module.__init__(model)
    model.config = type("Config", (), {"action_condition_scale": 1.0})()
    model.action_condition_proj = nn.Sequential(
        nn.LayerNorm(3),
        nn.Linear(3, 4),
        nn.SiLU(),
        nn.Linear(4, 8),
    )
    nn.init.zeros_(model.action_condition_proj[-1].weight)
    nn.init.zeros_(model.action_condition_proj[-1].bias)
    suffix = torch.randn(2, 5, 4)
    logits = torch.randn(2, 3, requires_grad=True)

    conditioned, metrics = model._condition_flow_hidden(suffix, logits)
    assert torch.equal(conditioned, suffix)
    assert metrics["action_condition_gamma_rms"].item() == 0
    assert metrics["action_condition_beta_rms"].item() == 0

    # Flow trains the condition projection, but the logits/VLM branch is trained
    # separately by its KL objective.
    conditioned.square().mean().backward()
    assert model.action_condition_proj[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.action_condition_proj[-1].weight.fill_(0.01)
    logits = torch.randn(2, 3, requires_grad=True)
    conditioned, _ = model._condition_flow_hidden(suffix, logits)
    conditioned.square().mean().backward()
    assert logits.grad is None


class _StageModel(SmolActionMemFlowMatching):
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
        self.action_condition_proj = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 4))
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
    assert all(
        parameter.requires_grad is expert_trainable for parameter in model.action_condition_proj.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.action_code_embedding.parameters())
    assert not any(parameter.requires_grad for parameter in model.vlm_with_expert.vlm.lm_head.parameters())


def test_default_peft_targets_do_not_train_language_vocabulary():
    policy = SmolActionMemPolicy.__new__(SmolActionMemPolicy)
    defaults = policy._get_default_peft_targets()
    pattern = defaults["target_modules"]

    assert re.fullmatch(
        pattern,
        "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj",
    )
    assert not re.fullmatch(pattern, "model.vlm_with_expert.vlm.lm_head")
    assert not re.fullmatch(pattern, "model.vlm_with_expert.vlm.model.text_model.embed_tokens")
    assert defaults["modules_to_save"] == [
        "action_code_embedding",
        "action_classifier",
        "action_condition_proj",
    ]
