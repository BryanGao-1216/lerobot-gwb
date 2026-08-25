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

"""Shared independent action-code protocol and objectives for ActionMem policies."""

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION_TOKEN_DISTANCES


@dataclass(frozen=True)
class ActionCodeLayout:
    """Model-local IDs; none of these IDs enter the language tokenizer vocabulary."""

    codebook_size: int = 256
    invalid_value: int = -1

    def __post_init__(self) -> None:
        if self.codebook_size < 2:
            raise ValueError(f"codebook_size must be at least 2, got {self.codebook_size}.")
        if 0 <= self.invalid_value < self.codebook_size:
            raise ValueError(
                f"invalid_value={self.invalid_value} overlaps valid action codes "
                f"[0, {self.codebook_size - 1}]."
            )

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

    @property
    def context_size(self) -> int:
        return self.codebook_size + 4


def fill_missing_initialized_state(
    state_dict: MutableMapping[str, Tensor],
    model: nn.Module,
    prefixes: Sequence[str],
) -> list[str]:
    """Fill new module keys so strict base-checkpoint loading remains possible."""
    initialized: list[str] = []
    for key, value in model.state_dict().items():
        if key not in state_dict and key.startswith(tuple(prefixes)):
            state_dict[key] = value.detach().clone()
            initialized.append(key)
    return initialized


def validate_action_code_sequence(
    action_tokens: Tensor | None,
    action_token_masks: Tensor | None,
    layout: ActionCodeLayout,
    *,
    policy_name: str,
) -> None:
    if action_tokens is None or action_token_masks is None:
        raise ValueError(f"{policy_name} requires action_tokens and action_token_masks.")
    if action_tokens.shape != action_token_masks.shape or action_tokens.ndim != 2:
        raise ValueError(
            "action_tokens and action_token_masks must have shape [B, T], got "
            f"{tuple(action_tokens.shape)} and {tuple(action_token_masks.shape)}."
        )
    if action_tokens.shape[1] < 2:
        raise ValueError(f"{policy_name} requires ACTION_QUERY plus a target slot.")

    invalid_query = (~action_token_masks[:, -2].bool()) | (
        action_tokens[:, -2] != layout.action_query_id
    )
    if torch.any(invalid_query):
        raise ValueError(
            f"The penultimate action token must be the valid ACTION_QUERY token {layout.action_query_id}."
        )

    targets = action_tokens[:, -1]
    target_masks = action_token_masks[:, -1].bool()
    invalid_targets = target_masks & ((targets < 0) | (targets >= layout.codebook_size))
    if torch.any(invalid_targets):
        invalid_ids = targets[invalid_targets].detach().cpu().tolist()
        raise ValueError(
            f"Action-code targets must be in [0, {layout.codebook_size - 1}], got {invalid_ids}."
        )


def compute_action_code_objective(
    logits: Tensor,
    action_tokens: Tensor,
    action_token_masks: Tensor,
    action_code_distances: Tensor | None,
    *,
    temperature: float,
    policy_name: str,
) -> dict[str, Tensor]:
    """Match predicted code logits to effect-tokenizer prototype-distance soft labels."""
    target_mask = action_token_masks[:, -1].bool()
    safe_targets = torch.where(target_mask, action_tokens[:, -1], 0)
    if torch.any(target_mask):
        if action_code_distances is None:
            raise ValueError(
                f"{policy_name} soft action-code training requires '{ACTION_TOKEN_DISTANCES}' in the "
                "processed batch. Use the RLDS effect-tokenizer collator or provide distances from "
                "the frozen effect latent to every codebook center."
            )
        if action_code_distances.shape != logits.shape:
            raise ValueError(
                f"Expected action-code distances with shape {tuple(logits.shape)}, got "
                f"{tuple(action_code_distances.shape)}."
            )
        distances = action_code_distances.to(device=logits.device, dtype=torch.float32)
        if not torch.isfinite(distances).all():
            raise ValueError("Action-code latent distances must contain only finite values.")
        soft_targets = torch.softmax(-distances / temperature, dim=-1)
        log_predictions = F.log_softmax(logits.float(), dim=-1)
        per_sample = F.kl_div(log_predictions, soft_targets, reduction="none").sum(dim=-1)
        target_entropy_per_sample = -(
            soft_targets * soft_targets.clamp_min(torch.finfo(soft_targets.dtype).tiny).log()
        ).sum(dim=-1)
        target_peak_per_sample = soft_targets.max(dim=-1).values
    else:
        per_sample = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
        target_entropy_per_sample = torch.zeros_like(per_sample)
        target_peak_per_sample = torch.zeros_like(per_sample)

    mask = target_mask.to(dtype=per_sample.dtype)
    valid_count = mask.sum().clamp_min(1)
    per_sample = per_sample * mask
    mean_loss = per_sample.sum() / valid_count
    accuracy = ((logits.argmax(dim=-1) == safe_targets) & target_mask).float().sum() / valid_count

    target_logits = logits.float().gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    greater_count = (logits.float() > target_logits.unsqueeze(-1)).sum(dim=-1)
    equal_count = (logits.float() == target_logits.unsqueeze(-1)).sum(dim=-1)
    target_rank_per_sample = greater_count.float() + (equal_count.float() + 1.0) / 2.0
    target_rank = (target_rank_per_sample * mask).sum() / valid_count
    target_entropy = (target_entropy_per_sample * mask).sum() / valid_count
    target_peak_probability = (target_peak_per_sample * mask).sum() / valid_count
    return {
        "action_token_loss_per_sample": per_sample,
        "action_token_target_mask": target_mask,
        "action_token_kl_loss": mean_loss,
        "action_token_accuracy": accuracy,
        "action_token_target_rank": target_rank,
        "action_token_soft_target_entropy": target_entropy,
        "action_token_soft_target_peak_probability": target_peak_probability,
    }


def condition_flow_hidden(
    suffix_out: Tensor,
    action_logits: Tensor,
    projection: nn.Module,
    *,
    scale: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply bounded logit-conditioned FiLM while blocking flow gradients into the VLM logits."""
    action_logits = action_logits.detach()
    projection_dtype = next(projection.parameters()).dtype
    film = torch.tanh(projection(action_logits.to(dtype=projection_dtype))).float()
    gamma, beta = film.chunk(2, dim=-1)
    gamma = gamma * scale
    beta = beta * scale
    conditioned = suffix_out.float() * (1.0 + gamma[:, None, :]) + beta[:, None, :]
    probabilities = torch.softmax(action_logits.float(), dim=-1)
    predicted_entropy = -(
        probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    ).sum(dim=-1)
    return conditioned, {
        "action_condition_gamma_rms": gamma.float().square().mean().sqrt(),
        "action_condition_beta_rms": beta.float().square().mean().sqrt(),
        "action_condition_logit_std": action_logits.float().std(dim=-1, unbiased=False).mean(),
        "action_condition_predicted_entropy": predicted_entropy.mean(),
    }
