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

"""Action-convention adapters used at the LIBERO environment boundary."""

from __future__ import annotations

import numpy as np

LIBERO_GRIPPER_ACTION_CONVENTIONS = ("libero", "oxe")


def validate_libero_gripper_action_convention(convention: str) -> None:
    if convention not in LIBERO_GRIPPER_ACTION_CONVENTIONS:
        raise ValueError(
            f"gripper_action_convention must be one of "
            f"{LIBERO_GRIPPER_ACTION_CONVENTIONS}, got {convention!r}."
        )


def convert_action_to_libero_gripper_convention(
    action: np.ndarray,
    *,
    source_convention: str,
) -> np.ndarray:
    """Convert only the final gripper dimension to LIBERO's actuator convention.

    ``libero`` actions already use ``-1=open, +1=close`` and pass through.
    ``oxe`` actions use ``0=close, 1=open``.  Match the OpenVLA/VQ-VLA
    evaluation contract by mapping them to ``[-1, +1]``, binarizing with
    ``sign``, and then inverting the sign for LIBERO.  In one expression this
    is ``-sign(2 * gripper - 1)``.
    """
    validate_libero_gripper_action_convention(source_convention)
    values = np.asarray(action)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Expected a non-empty 1-D action, got shape {values.shape}.")

    converted = np.array(values, copy=True)
    if source_convention == "libero":
        return converted

    gripper = float(converted[-1])
    if not np.isfinite(gripper):
        raise ValueError(f"OXE gripper action must be finite, got {gripper}.")
    converted[-1] = -np.sign(2.0 * gripper - 1.0)
    return converted
