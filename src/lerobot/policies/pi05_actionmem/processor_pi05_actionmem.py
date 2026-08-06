#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .configuration_pi05_actionmem import PI05ActionMemConfig
from .tokenization_pi05_actionmem import PI05ActionMemTokenMap, default_action_token_map_path


@ProcessorStepRegistry.register(name="pi05_actionmem_prepare_state_tokenizer_processor_step")
@dataclass
class PI05ActionMemPrepareStateTokenizerProcessorStep(ProcessorStep):
    """Build the native PI0.5 task/state/action text prompt."""

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05ActionMem")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # NormalizerProcessorStep has already mapped state to the PI0.5
        # quantile range. Keep the native openpi-compatible 256-bin text form.
        state_np = deepcopy(state).detach().cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for index, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[index]))
            full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.

        Args:
            features: The input feature dictionary.

        Returns:
            The unchanged feature dictionary.
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="pi05_actionmem_action_token_processor")
class PI05ActionMemActionTokenProcessorStep(ComplementaryDataProcessorStep):
    """Convert a per-frame q0 code into PaliGemma action-token inputs.

    The history slot is represented by an empty, delimited memory block:
    ``[MEMORY_START, MEMORY_END, ACTION_QUERY, CURRENT_ACTION_TOKEN]``.
    The final target position is padded during inference or when the dataset
    stores the configured invalid value.
    """

    token_map_path: str
    action_token_key: str = ACTION_TOKEN
    _token_map: PI05ActionMemTokenMap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._token_map = PI05ActionMemTokenMap.from_json(self.token_map_path)
        self.token_map_path = self._token_map.path

    def _batch_size_and_device(self) -> tuple[int, torch.device]:
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        language_tokens = observation.get(OBS_LANGUAGE_TOKENS)
        if not isinstance(language_tokens, torch.Tensor) or language_tokens.ndim < 1:
            raise ValueError(
                "PI05ActionMemActionTokenProcessorStep must run after TokenizerProcessorStep so that "
                f"'{OBS_LANGUAGE_TOKENS}' is available."
            )
        return language_tokens.shape[0], language_tokens.device

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        batch_size, device = self._batch_size_and_device()

        tokens = torch.full(
            (batch_size, 4),
            self._token_map.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        masks = torch.zeros((batch_size, 4), dtype=torch.bool, device=device)
        tokens[:, 0] = self._token_map.memory_start_token_id
        tokens[:, 1] = self._token_map.memory_end_token_id
        tokens[:, 2] = self._token_map.action_query_token_id
        masks[:, :3] = True

        raw_q0 = complementary_data.get(self.action_token_key)
        if raw_q0 is not None:
            raw_q0_tensor = torch.as_tensor(raw_q0)
            q0 = raw_q0_tensor.to(device=device, dtype=torch.long).reshape(-1)
            if q0.numel() != batch_size:
                raise ValueError(
                    f"Expected {batch_size} q0 codes in '{self.action_token_key}', got shape "
                    f"{tuple(raw_q0_tensor.shape)}."
                )

            valid = q0 != self._token_map.invalid_value
            out_of_range = valid & ((q0 < self._token_map.code_id_min) | (q0 > self._token_map.code_id_max))
            if torch.any(out_of_range):
                invalid_codes = q0[out_of_range].detach().cpu().tolist()
                raise ValueError(
                    f"PI05ActionMem q0 codes must be in "
                    f"[{self._token_map.code_id_min}, {self._token_map.code_id_max}] "
                    f"or equal invalid_value={self._token_map.invalid_value}; got {invalid_codes}."
                )

            tokens[valid, 3] = self._token_map.anchor_token_id - q0[valid]
            masks[valid, 3] = True

        complementary_data[ACTION_TOKENS] = tokens
        complementary_data[ACTION_TOKEN_MASK] = masks
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "token_map_path": self.token_map_path,
            "action_token_key": self.action_token_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi05_actionmem_pre_post_processors(
    config: PI05ActionMemConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI05ActionMem policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Mapping the dataset q0 code to the PI05ActionMem action-token protocol.
    7. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI05ActionMem policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # PI0.5 order: raw → relative → normalize → state prompt →
    # tokenize → ActionMem protocol → model.
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,
        steps.normalize,
        PI05ActionMemPrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        PI05ActionMemActionTokenProcessorStep(
            token_map_path=config.action_token_map_path or str(default_action_token_map_path()),
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def reconcile_pi05_actionmem_processors(
    config: PI05ActionMemConfig,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Upgrade a copied PI0.5 processor artifact with ActionMem token handling.

    A PI05ActionMem base checkpoint is commonly produced by copying a PI0.5
    artifact. Its saved normalizer and PI0.5 state-prompt steps remain valid,
    but the action-token step is absent. Insert it immediately after language
    tokenization without disturbing the saved dataset-stat overrides.
    """

    steps = list(preprocessor.steps)
    existing_index = next(
        (
            index
            for index, step in enumerate(steps)
            if isinstance(step, PI05ActionMemActionTokenProcessorStep)
        ),
        None,
    )
    token_map_path = config.action_token_map_path or str(default_action_token_map_path())
    replacement = PI05ActionMemActionTokenProcessorStep(token_map_path=token_map_path)
    if existing_index is not None:
        steps[existing_index] = replacement
        preprocessor.steps = steps
        return preprocessor, postprocessor

    tokenizer_index = next(
        (index for index, step in enumerate(steps) if isinstance(step, TokenizerProcessorStep)),
        None,
    )
    if tokenizer_index is None:
        raise ValueError("Cannot reconcile PI05ActionMem preprocessor: TokenizerProcessorStep is missing.")
    steps.insert(tokenizer_index + 1, replacement)
    preprocessor.steps = steps
    return preprocessor, postprocessor
