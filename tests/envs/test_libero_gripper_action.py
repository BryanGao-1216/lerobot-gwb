#!/usr/bin/env python

from __future__ import annotations

import numpy as np
import pytest

from lerobot.envs.libero_utils import convert_action_to_libero_gripper_convention


@pytest.mark.parametrize(
    ("oxe_gripper", "libero_gripper"),
    [
        (0.0, 1.0),
        (0.49, 1.0),
        (0.5, 0.0),
        (0.51, -1.0),
        (1.0, -1.0),
    ],
)
def test_oxe_gripper_is_binarized_and_inverted_for_libero(oxe_gripper, libero_gripper):
    action = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, oxe_gripper], dtype=np.float32)
    original = action.copy()

    converted = convert_action_to_libero_gripper_convention(action, source_convention="oxe")

    np.testing.assert_array_equal(converted[:6], action[:6])
    assert converted[-1] == libero_gripper
    np.testing.assert_array_equal(action, original)


def test_native_libero_action_is_unchanged_and_copied():
    action = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0], dtype=np.float32)

    converted = convert_action_to_libero_gripper_convention(action, source_convention="libero")

    np.testing.assert_array_equal(converted, action)
    assert converted is not action


def test_unknown_gripper_action_convention_is_rejected():
    with pytest.raises(ValueError, match="gripper_action_convention"):
        convert_action_to_libero_gripper_convention(
            np.zeros(7, dtype=np.float32),
            source_convention="unknown",
        )


def test_non_finite_oxe_gripper_is_rejected():
    action = np.zeros(7, dtype=np.float32)
    action[-1] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        convert_action_to_libero_gripper_convention(action, source_convention="oxe")
