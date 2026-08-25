import re

import pytest
import torch
from torch import nn

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pi05_actionmem.configuration_pi05_actionmem import PI05ActionMemConfig
from lerobot.policies.pi05_actionmem.modeling_pi05_actionmem import (
    PI05ActionMemPolicy,
    PI05ActionMemPytorch,
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
    ACTION_TOKEN_DISTANCES,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def test_policy_is_registered_and_keeps_native_pi05_normalization():
    config = PI05ActionMemConfig()

    assert config.type == "pi05_actionmem"
    assert config.chunk_size == 10
    assert config.tokenizer_max_length == 200
    assert config.normalization_mapping["STATE"] is NormalizationMode.QUANTILES
    assert config.normalization_mapping["ACTION"] is NormalizationMode.QUANTILES
    assert PreTrainedConfig.get_choice_class("pi05_actionmem") is PI05ActionMemConfig
    assert get_policy_class("pi05_actionmem") is PI05ActionMemPolicy


def test_action_code_processor_uses_independent_local_ids():
    step = PI05ActionMemActionTokenProcessorStep(codebook_size=256, invalid_value=-1)
    transition = create_transition(
        observation={OBS_LANGUAGE_TOKENS: torch.ones(2, 4, dtype=torch.long)},
        complementary_data={"task": ["pick", "place"], ACTION_TOKEN: torch.tensor([[0], [255]])},
    )

    output = step(transition)[TransitionKey.COMPLEMENTARY_DATA]

    assert torch.equal(
        output[ACTION_TOKENS],
        torch.tensor([[256, 257, 258, 0], [256, 257, 258, 255]]),
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


def test_copied_pi05_processor_is_upgraded_with_action_codes():
    tokenizer_step = TokenizerProcessorStep.__new__(TokenizerProcessorStep)
    preprocessor = DataProcessorPipeline(steps=(tokenizer_step,))
    postprocessor = DataProcessorPipeline(steps=())
    config = PI05ActionMemConfig()

    reconciled, _ = reconcile_pi05_actionmem_processors(config, preprocessor, postprocessor)

    assert reconciled.steps[0] is tokenizer_step
    assert isinstance(reconciled.steps[1], PI05ActionMemActionTokenProcessorStep)
    assert reconciled.steps[1].codebook_size == 256


def test_pi05_suffix_uses_time_adarms_without_a_state_projection():
    model = PI05ActionMemPytorch.__new__(PI05ActionMemPytorch)
    nn.Module.__init__(model)
    model.config = type(
        "Config", (), {"chunk_size": 10, "min_period": 4e-3, "max_period": 4.0}
    )()
    model.action_in_proj = nn.Linear(2, 4)
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model._apply_checkpoint = lambda function, *args, **kwargs: function(*args, **kwargs)

    embeddings, masks, attention, adarms = model.embed_suffix(
        noisy_actions=torch.zeros(2, 10, 2),
        timestep=torch.tensor([0.25, 0.75]),
    )

    assert embeddings.shape == (2, 10, 4)
    assert masks.shape == (2, 10)
    assert attention.shape == (2, 10)
    assert adarms.shape == (2, 4)
    assert not hasattr(model, "state_proj")


class _DummyTrainingCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.distances = []

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
        action_token_distances,
        compute_flow,
        compute_action_token,
    ):
        del images, image_masks, language_tokens, language_masks, action_tokens, action_token_masks
        del noise, time
        self.calls.append((compute_flow, compute_action_token))
        self.distances.append(action_token_distances)
        device = actions.device if actions is not None else action_token_distances.device
        output = {}
        if compute_flow:
            output.update(
                {
                    "flow_losses": torch.full_like(actions, 2.0),
                    "action_condition_gamma_rms": torch.tensor(0.0, device=device),
                    "action_condition_beta_rms": torch.tensor(0.0, device=device),
                    "action_condition_logit_std": torch.tensor(1.0, device=device),
                    "action_condition_predicted_entropy": torch.tensor(2.0, device=device),
                }
            )
        if compute_action_token:
            output.update(
                {
                    "action_token_loss_per_sample": torch.tensor([3.0, 0.0], device=device),
                    "action_token_target_mask": torch.tensor([True, False], device=device),
                    "action_token_kl_loss": torch.tensor(3.0, device=device),
                    "action_token_accuracy": torch.tensor(0.5, device=device),
                    "action_token_target_rank": torch.tensor(4.0, device=device),
                    "action_token_soft_target_entropy": torch.tensor(1.5, device=device),
                    "action_token_soft_target_peak_probability": torch.tensor(0.4, device=device),
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
def test_policy_training_stages_select_effect_code_and_flow_objectives(
    stage, expected_scalar, expected_per_sample, expected_call
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
    distances = torch.rand(2, 256)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION_TOKENS: torch.tensor([[256, 257, 258, 10], [256, 257, 258, 259]]),
        ACTION_TOKEN_MASK: torch.tensor([[True, True, True, True], [True, True, True, False]]),
        ACTION_TOKEN_DISTANCES: distances,
        ACTION: torch.zeros(2, 1, 2),
    }

    scalar_loss, metrics = policy.forward(batch)
    per_sample_loss, _ = policy.forward(batch, reduction="none")

    assert scalar_loss.item() == expected_scalar
    assert torch.equal(per_sample_loss, expected_per_sample)
    assert policy.model.calls == [expected_call, expected_call]
    assert all(value is distances for value in policy.model.distances)
    if stage != "action_expert_only":
        assert metrics["action_token_kl_loss"] == 3.0


def test_training_stage_freezes_new_pi05_action_modules():
    class StageModel(PI05ActionMemPytorch):
        def __init__(self, stage):
            nn.Module.__init__(self)
            self.config = type("Config", (), {"training_stage": stage})()
            self.paligemma_with_expert = nn.Module()
            self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
            self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
            self.action_code_embedding = nn.Embedding(8, 2)
            self.action_classifier = nn.Linear(2, 4)
            self.action_condition_proj = nn.Linear(4, 4)
            self.action_in_proj = nn.Linear(2, 2)
            self.action_out_proj = nn.Linear(2, 2)
            self.time_mlp_in = nn.Linear(2, 2)
            self.time_mlp_out = nn.Linear(2, 2)

    vlm = StageModel("vlm_only")
    vlm.configure_training_stage()
    assert vlm.action_classifier.weight.requires_grad
    assert not vlm.action_condition_proj.weight.requires_grad

    expert = StageModel("action_expert_only")
    expert.configure_training_stage()
    assert not expert.action_classifier.weight.requires_grad
    assert expert.action_condition_proj.weight.requires_grad


@pytest.mark.parametrize("stage", ["vlm_only", "action_expert_only", "joint"])
def test_default_peft_targets_save_independent_action_modules(stage):
    policy = PI05ActionMemPolicy.__new__(PI05ActionMemPolicy)
    policy.config = type("Config", (), {"training_stage": stage})()
    targets = policy._get_default_peft_targets()

    assert set(targets["modules_to_save"]) == {
        "action_code_embedding",
        "action_classifier",
        "action_condition_proj",
    }
    assert bool(re.fullmatch(targets["target_modules"], "model.time_mlp_in")) is (
        stage != "vlm_only"
    )
