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

from dataclasses import dataclass
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

from .configuration_smol_actionmem import SmolActionMemConfig


@dataclass
@ProcessorStepRegistry.register(name="smol_actionmem_action_code_processor")
class SmolActionMemActionCodeProcessorStep(ComplementaryDataProcessorStep):
    """Place each endpoint-effect code in the independent action context.

    History is intentionally empty in this first version:
    ``[MEMORY_START, MEMORY_END, ACTION_QUERY, CURRENT_TARGET]``.
    These IDs index a model-local embedding table and never enter the SmolVLM
    tokenizer or language embedding matrix.
    """

    codebook_size: int = 256
    invalid_value: int = -1
    action_token_key: str = ACTION_TOKEN

    def __post_init__(self) -> None:
        if self.codebook_size < 2:
            raise ValueError(f"codebook_size must be at least 2, got {self.codebook_size}.")

    @property
    def memory_start_id(self) -> int:
        return self.codebook_size

    @property
    def memory_end_id(self) -> int:
        return self.codebook_size + 1

    @property
    def action_query_id(self) -> int:
        return self.codebook_size + 2

    @property
    def padding_id(self) -> int:
        return self.codebook_size + 3

    def _batch_size_and_device(self) -> tuple[int, torch.device]:
        observation = self.transition.get(TransitionKey.OBSERVATION) or {}
        language_tokens = observation.get(OBS_LANGUAGE_TOKENS)
        if not isinstance(language_tokens, torch.Tensor) or language_tokens.ndim < 1:
            raise ValueError(
                "SmolActionMemActionCodeProcessorStep must run after TokenizerProcessorStep so that "
                f"'{OBS_LANGUAGE_TOKENS}' is available."
            )
        return language_tokens.shape[0], language_tokens.device

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        batch_size, device = self._batch_size_and_device()
        tokens = torch.full(
            (batch_size, 4),
            self.padding_id,
            dtype=torch.long,
            device=device,
        )
        masks = torch.zeros((batch_size, 4), dtype=torch.bool, device=device)
        tokens[:, 0] = self.memory_start_id
        tokens[:, 1] = self.memory_end_id
        tokens[:, 2] = self.action_query_id
        masks[:, :3] = True

        raw_codes = complementary_data.get(self.action_token_key)
        if raw_codes is not None:
            raw_code_tensor = torch.as_tensor(raw_codes)
            codes = raw_code_tensor.to(device=device, dtype=torch.long).reshape(-1)
            if codes.numel() != batch_size:
                raise ValueError(
                    f"Expected {batch_size} action codes in '{self.action_token_key}', got shape "
                    f"{tuple(raw_code_tensor.shape)}."
                )

            valid = codes != self.invalid_value
            out_of_range = valid & ((codes < 0) | (codes >= self.codebook_size))
            if torch.any(out_of_range):
                invalid_codes = codes[out_of_range].detach().cpu().tolist()
                raise ValueError(
                    f"Smol ActionMem action codes must be in [0, {self.codebook_size - 1}] "
                    f"or equal invalid_value={self.invalid_value}; got {invalid_codes}."
                )

            tokens[valid, 3] = codes[valid]
            masks[valid, 3] = True

        complementary_data[ACTION_TOKENS] = tokens
        complementary_data[ACTION_TOKEN_MASK] = masks
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "codebook_size": self.codebook_size,
            "invalid_value": self.invalid_value,
            "action_token_key": self.action_token_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_smol_actionmem_pre_post_processors(
    config: SmolActionMemConfig,
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
        SmolActionMemActionCodeProcessorStep(
            codebook_size=config.action_codebook_size,
            invalid_value=config.action_code_invalid_value,
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
