#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import json
from dataclasses import dataclass
from pathlib import Path


def default_action_token_map_path() -> Path:
    """Return the workspace-local ActionMem token-map path."""
    return Path(__file__).resolve().parents[6] / "models" / "actionmem-base" / "token_map.json"


@dataclass(frozen=True)
class ActionMemTokenMap:
    """Validated token IDs shared by the ActionMem processor and model."""

    path: str
    vqvae_checkpoint_path: str | None
    codebook_size: int
    code_id_min: int
    code_id_max: int
    invalid_value: int
    action_horizon: int
    action_dim: int
    anchor_token_id: int
    token_id_min: int
    token_id_max: int
    memory_start_token_id: int
    memory_end_token_id: int
    action_query_token_id: int
    pad_token_id: int

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "ActionMemTokenMap":
        resolved_path = Path(path or default_action_token_map_path()).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"ActionMem token map does not exist: {resolved_path}")

        with resolved_path.open(encoding="utf-8") as file:
            raw_map = json.load(file)

        vqvae = raw_map["vqvae"]
        action_tokens = raw_map["action_tokens"]
        control_tokens = raw_map["control_tokens"]
        padding = raw_map["padding"]

        checkpoint_path = vqvae.get("checkpoint_path")
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = resolved_path.parent / checkpoint_path
            checkpoint_path = str(checkpoint_path.resolve())

        token_map = cls(
            path=str(resolved_path),
            vqvae_checkpoint_path=checkpoint_path,
            codebook_size=int(vqvae["codebook_size"]),
            code_id_min=int(vqvae["code_id_min"]),
            code_id_max=int(vqvae["code_id_max"]),
            invalid_value=int(vqvae.get("invalid_value", -1)),
            action_horizon=int(vqvae["action_horizon"]),
            action_dim=int(vqvae["action_dim"]),
            anchor_token_id=int(action_tokens["anchor_token_id"]),
            token_id_min=int(action_tokens["token_id_min"]),
            token_id_max=int(action_tokens["token_id_max"]),
            memory_start_token_id=int(control_tokens["action_memory_start"]["token_id"]),
            memory_end_token_id=int(control_tokens["action_memory_end"]["token_id"]),
            action_query_token_id=int(control_tokens["action_query"]["token_id"]),
            pad_token_id=int(padding["token_id"]),
        )
        token_map._validate()
        return token_map

    def _validate(self) -> None:
        expected_codebook_size = self.code_id_max - self.code_id_min + 1
        if self.codebook_size != expected_codebook_size:
            raise ValueError(
                "Invalid ActionMem token map: codebook_size does not match the configured code ID range "
                f"({self.codebook_size} != {expected_codebook_size})."
            )

        mapped_min = self.anchor_token_id - self.code_id_max
        mapped_max = self.anchor_token_id - self.code_id_min
        if mapped_min != self.token_id_min or mapped_max != self.token_id_max:
            raise ValueError(
                "Invalid ActionMem token map: q0 mapping formula does not match token ID bounds."
            )

        control_ids = {
            self.memory_start_token_id,
            self.memory_end_token_id,
            self.action_query_token_id,
        }
        if len(control_ids) != 3:
            raise ValueError("Invalid ActionMem token map: control token IDs must be distinct.")

    def q0_to_token_id(self, q0_code: int) -> int:
        if q0_code < self.code_id_min or q0_code > self.code_id_max:
            raise ValueError(f"q0 code must be in [{self.code_id_min}, {self.code_id_max}], got {q0_code}.")
        return self.anchor_token_id - q0_code

    def token_id_to_q0(self, token_id: int) -> int:
        if token_id < self.token_id_min or token_id > self.token_id_max:
            raise ValueError(
                f"Action token ID must be in [{self.token_id_min}, {self.token_id_max}], got {token_id}."
            )
        return self.anchor_token_id - token_id
