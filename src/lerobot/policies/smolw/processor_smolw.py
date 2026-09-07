#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SmolW preprocessing for terminal video and action padding."""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import EnvTransition, TransitionKey

from .configuration_smolw import SmolWConfig


@dataclass
@ProcessorStepRegistry.register(name="smolw_stationary_action_padding")
class SmolWStationaryActionPaddingProcessorStep(ProcessorStep):
    """Replace episode-tail action padding with valid stationary commands.

    This step runs before action normalization. For native LIBERO actions the
    six pose-delta dimensions become zero while the absolute gripper command
    retains its last in-episode value. The synthetic commands are deliberately
    marked valid so that they participate in action flow matching.
    """

    hold_dims: list[int]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if action is None or complementary_data is None or "action_is_pad" not in complementary_data:
            return new_transition
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"SmolW stationary action padding expects a tensor, got {type(action).__name__}.")
        if action.ndim < 2:
            raise ValueError(
                "SmolW stationary action padding expects [..., horizon, action_dim], "
                f"got {tuple(action.shape)}."
            )

        is_pad = torch.as_tensor(
            complementary_data["action_is_pad"],
            dtype=torch.bool,
            device=action.device,
        )
        if is_pad.shape != action.shape[:-1]:
            raise ValueError(
                "action_is_pad must match the action horizon dimensions; got "
                f"{tuple(is_pad.shape)} for actions {tuple(action.shape)}."
            )
        if not torch.any(is_pad):
            return new_transition

        horizon, action_dim = action.shape[-2:]
        resolved_hold_dims: list[int] = []
        for index in self.hold_dims:
            resolved = index if index >= 0 else action_dim + index
            if resolved < 0 or resolved >= action_dim:
                raise ValueError(
                    f"stationary action hold index {index} is invalid for action_dim={action_dim}."
                )
            resolved_hold_dims.append(resolved)

        flat_action = action.reshape(-1, horizon, action_dim)
        flat_is_pad = is_pad.reshape(-1, horizon)
        positions = torch.arange(horizon, device=action.device).expand_as(flat_is_pad)
        last_valid_positions = positions.masked_fill(flat_is_pad, -1).max(dim=1).values
        # A fully padded row is not produced by the episode-aware sampler, but
        # falling back to the reader's clamped final action keeps this step total.
        last_valid_positions = torch.where(
            last_valid_positions >= 0,
            last_valid_positions,
            torch.full_like(last_valid_positions, horizon - 1),
        )
        row_indices = torch.arange(flat_action.shape[0], device=action.device)
        last_valid_actions = flat_action[row_indices, last_valid_positions]

        stationary_actions = torch.zeros_like(last_valid_actions)
        if resolved_hold_dims:
            stationary_actions[:, resolved_hold_dims] = last_valid_actions[:, resolved_hold_dims]
        flat_action = torch.where(
            flat_is_pad.unsqueeze(-1),
            stationary_actions.unsqueeze(1),
            flat_action,
        )

        new_transition[TransitionKey.ACTION] = flat_action.reshape_as(action)
        new_complementary_data = complementary_data.copy()
        new_complementary_data["action_is_pad"] = torch.zeros_like(is_pad)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"hold_dims": list(self.hold_dims)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_smolw_pre_post_processors(
    config: SmolWConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build SmolVLA preprocessing plus raw-space stationary tail actions."""
    steps = make_default_policy_processor_steps(config, dataset_stats)
    input_steps = [
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
        SmolWStationaryActionPaddingProcessorStep(
            hold_dims=config.stationary_action_hold_dims,
        ),
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def reconcile_smolw_processors(
    config: SmolWConfig,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Add or refresh SmolW tail handling in copied SmolVLA processors."""

    steps = list(preprocessor.steps)
    replacement = SmolWStationaryActionPaddingProcessorStep(
        hold_dims=config.stationary_action_hold_dims,
    )
    existing_index = next(
        (
            index
            for index, step in enumerate(steps)
            if isinstance(step, SmolWStationaryActionPaddingProcessorStep)
        ),
        None,
    )
    if existing_index is not None:
        steps[existing_index] = replacement
        preprocessor.steps = steps
        return preprocessor, postprocessor

    normalizer_index = next(
        (index for index, step in enumerate(steps) if isinstance(step, NormalizerProcessorStep)),
        None,
    )
    if normalizer_index is None:
        raise ValueError("Cannot reconcile SmolW preprocessor: NormalizerProcessorStep is missing.")
    steps.insert(normalizer_index, replacement)
    preprocessor.steps = steps
    return preprocessor, postprocessor
