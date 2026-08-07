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

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ComplementaryDataProcessorStep,
    NewLineTaskProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION_TOKEN, ACTION_TOKEN_MASK, ACTION_TOKENS, OBS_LANGUAGE_TOKENS

from .configuration_smol_actionmem import SmolActionMem2Config
from .tokenization_smol_actionmem import (
    SmolActionMem2TokenMap,
    default_smol_actionmem2_token_map_path,
)


@dataclass
@ProcessorStepRegistry.register(name="smol_actionmem2_action_code_processor")
class SmolActionMem2ActionCodeProcessorStep(ComplementaryDataProcessorStep):
    """Map each frame's q0 code to the independent action context.

    History is intentionally empty in this first version:
    ``[MEMORY_START, MEMORY_END, ACTION_QUERY, CURRENT_TARGET]``.
    These IDs index a model-local embedding table and never enter the SmolVLM
    tokenizer or language embedding matrix.
    """

    token_map_path: str
    action_token_key: str = ACTION_TOKEN
    _token_map: SmolActionMem2TokenMap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._token_map = SmolActionMem2TokenMap.from_json(self.token_map_path)
        self.token_map_path = self._token_map.path

    def _batch_size_and_device(self) -> tuple[int, torch.device]:
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        language_tokens = observation.get(OBS_LANGUAGE_TOKENS)
        if not isinstance(language_tokens, torch.Tensor) or language_tokens.ndim < 1:
            raise ValueError(
                "SmolActionMem2ActionCodeProcessorStep must run after TokenizerProcessorStep so that "
                f"'{OBS_LANGUAGE_TOKENS}' is available."
            )
        return language_tokens.shape[0], language_tokens.device

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        batch_size, device = self._batch_size_and_device()
        tokens = torch.full(
            (batch_size, 4),
            self._token_map.padding_id,
            dtype=torch.long,
            device=device,
        )
        masks = torch.zeros((batch_size, 4), dtype=torch.bool, device=device)
        tokens[:, 0] = self._token_map.memory_start_id
        tokens[:, 1] = self._token_map.memory_end_id
        tokens[:, 2] = self._token_map.action_query_id
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
                    f"Smol ActionMem 2 q0 codes must be in "
                    f"[{self._token_map.code_id_min}, {self._token_map.code_id_max}] "
                    f"or equal invalid_value={self._token_map.invalid_value}; got {invalid_codes}."
                )

            tokens[valid, 3] = q0[valid] - self._token_map.code_id_min
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


def make_smol_actionmem2_pre_post_processors(
    config: SmolActionMem2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name or config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        SmolActionMem2ActionCodeProcessorStep(
            token_map_path=config.action_token_map_path or str(default_smol_actionmem2_token_map_path()),
        ),
        steps.to_device,
        relative_step,
        steps.normalize,
    ]
    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
