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

"""VQ metadata and local action-code IDs for Smol ActionMem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def default_smol_actionmem_token_map_path() -> Path:
    return Path(__file__).resolve().parents[6] / "models" / "smol_actionmem-base" / "token_map.json"


@dataclass(frozen=True)
class SmolActionMemTokenMap:
    """Describe q0 codes without assigning them to the language vocabulary.

    Existing ActionMem token-map files remain valid: only their ``vqvae``
    section is consumed. Action and control IDs are allocated in a separate,
    model-local embedding table as ``[q0 classes, memory start, memory end,
    query, padding]``.
    """

    path: str
    vqvae_checkpoint_path: str | None
    codebook_size: int
    code_id_min: int
    code_id_max: int
    invalid_value: int
    action_horizon: int
    action_dim: int

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> SmolActionMemTokenMap:
        resolved_path = Path(path or default_smol_actionmem_token_map_path()).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Smol ActionMem token map does not exist: {resolved_path}")

        with resolved_path.open(encoding="utf-8") as file:
            raw_map = json.load(file)
        vqvae = raw_map["vqvae"]

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
        )
        token_map._validate()
        return token_map

    def _validate(self) -> None:
        expected_size = self.code_id_max - self.code_id_min + 1
        if self.codebook_size != expected_size:
            raise ValueError(
                "Invalid Smol ActionMem token map: codebook_size does not match the q0 range "
                f"({self.codebook_size} != {expected_size})."
            )
        if self.codebook_size <= 0:
            raise ValueError("Smol ActionMem requires a non-empty q0 codebook.")

    @property
    def action_class_min(self) -> int:
        return 0

    @property
    def action_class_max(self) -> int:
        return self.codebook_size - 1

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
    def context_vocab_size(self) -> int:
        return self.codebook_size + 4

    def q0_to_action_class(self, q0_code: int) -> int:
        if q0_code < self.code_id_min or q0_code > self.code_id_max:
            raise ValueError(f"q0 code must be in [{self.code_id_min}, {self.code_id_max}], got {q0_code}.")
        return q0_code - self.code_id_min

    def action_class_to_q0(self, action_class: int) -> int:
        if action_class < self.action_class_min or action_class > self.action_class_max:
            raise ValueError(
                f"Action class must be in [{self.action_class_min}, {self.action_class_max}], "
                f"got {action_class}."
            )
        return action_class + self.code_id_min
