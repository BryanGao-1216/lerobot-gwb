"""Native SmolVLA preprocessing with stationary episode-tail action targets."""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import EnvTransition, TransitionKey

from .configuration_smolvla_lap import SmolVLALAPConfig


@dataclass
@ProcessorStepRegistry.register(name="smolvla_lap_stationary_action_padding")
class LAPStationaryActionPaddingProcessorStep(ProcessorStep):
    """Pad 7D relative actions with [0,0,0,0,0,0,last_gripper], in raw units."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        extras = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        is_pad = extras.get("action_is_pad")
        if action is None or is_pad is None:
            return transition
        is_pad = is_pad.to(device=action.device, dtype=torch.bool)
        if action.ndim != 3 or is_pad.shape != action.shape[:-1]:
            raise ValueError("Expected actions [B,T,D] with action_is_pad [B,T].")
        if action.shape[-1] != 7:
            raise ValueError("LAP stationary tail padding requires 7D relative-pose + gripper actions.")
        # LeRobot clamps padded actions to the episode's last real action.
        stationary = torch.zeros_like(action)
        stationary[..., -1] = action[..., -1]
        result = transition.copy()
        result[TransitionKey.ACTION] = torch.where(is_pad[..., None], stationary, action)
        # Stationary commands are deliberate supervised targets, including at the tail.
        result[TransitionKey.COMPLEMENTARY_DATA] = {**extras, "action_is_pad": torch.zeros_like(is_pad)}
        return result

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_smolvla_lap_pre_post_processors(config: SmolVLALAPConfig, dataset_stats=None):
    steps = make_default_policy_processor_steps(config, dataset_stats)
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            NewLineTaskProcessorStep(),
            TokenizerProcessorStep(
                tokenizer_name=config.vlm_model_name,
                padding=config.pad_language_to,
                padding_side="right",
                max_length=config.tokenizer_max_length,
            ),
            steps.to_device,
            LAPStationaryActionPaddingProcessorStep(),
            steps.normalize,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )


def reconcile_smolvla_lap_processors(preprocessor, postprocessor):
    """Ensure saved pipelines also use raw-space tail padding when switching stages."""
    if not any(isinstance(step, LAPStationaryActionPaddingProcessorStep) for step in preprocessor.steps):
        index = next(
            i for i, step in enumerate(preprocessor.steps) if isinstance(step, NormalizerProcessorStep)
        )
        preprocessor.steps.insert(index, LAPStationaryActionPaddingProcessorStep())
    return preprocessor, postprocessor
