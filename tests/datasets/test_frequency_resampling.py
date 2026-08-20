import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RLDS_DIR = Path(__file__).parents[2] / "src" / "lerobot" / "datasets" / "rlds"
frequency_resampling = _load_module("frequency_resampling", RLDS_DIR / "frequency_resampling.py")
control_frequencies = _load_module("control_frequencies", RLDS_DIR / "oxe" / "control_frequencies.py")
nearest_frame_indices_numpy = frequency_resampling.nearest_frame_indices_numpy
resample_action_numpy = frequency_resampling.resample_action_numpy
resample_trajectory_tensor = frequency_resampling.resample_trajectory_tensor
get_oxe_control_frequency_hz = control_frequencies.get_oxe_control_frequency_hz


ABSOLUTE_GRIPPER = [False] * 6 + [True]


def _actions(relative_values, gripper_values):
    action = np.zeros((len(relative_values), 7), dtype=np.float32)
    action[:, 0] = relative_values
    action[:, -1] = gripper_values
    return action


def test_downsample_accumulates_relative_deltas_and_holds_gripper():
    source = _actions([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 0.0])
    result = resample_action_numpy(
        source,
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=20.0,
        target_hz=10.0,
    )

    assert result.shape == (2, 7)
    np.testing.assert_allclose(result[:, 0], [3.0, 7.0])
    np.testing.assert_allclose(result[:, -1], [1.0, 0.0])


def test_upsample_splits_relative_deltas_and_repeats_gripper():
    source = _actions([2.0, 4.0], [0.0, 1.0])
    result = resample_action_numpy(
        source,
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=5.0,
        target_hz=10.0,
    )

    assert result.shape == (4, 7)
    np.testing.assert_allclose(result[:, 0], [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(result[:, -1], [0.0, 0.0, 1.0, 1.0])


def test_nearest_frame_upsampling_reuses_nearest_source_image():
    assert nearest_frame_indices_numpy(3, source_hz=5.0, target_hz=10.0).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_tensor_trajectory_uses_one_nearest_index_for_image_state_and_language():
    tf = pytest.importorskip("tensorflow")
    source = {
        "action": tf.constant(_actions([2.0, 4.0], [0.0, 1.0])),
        "observation": {
            "image_primary": tf.constant([b"image-0", b"image-1"]),
            "proprio": tf.constant([[0.0], [2.0]], dtype=tf.float32),
            "timestep": tf.range(2),
        },
        "task": {"language_instruction": tf.constant([b"task-0", b"task-1"])},
        "dataset_name": tf.constant([b"toy", b"toy"]),
        "absolute_action_mask": tf.tile(tf.constant(ABSOLUTE_GRIPPER)[None], [2, 1]),
    }

    result = resample_trajectory_tensor(
        source,
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=5.0,
        target_hz=10.0,
    )

    assert result["observation"]["image_primary"].numpy().tolist() == [
        b"image-0",
        b"image-0",
        b"image-1",
        b"image-1",
    ]
    np.testing.assert_allclose(result["observation"]["proprio"], [[0.0], [0.0], [2.0], [2.0]])
    assert result["task"]["language_instruction"].numpy().tolist() == [
        b"task-0",
        b"task-0",
        b"task-1",
        b"task-1",
    ]
    np.testing.assert_array_equal(result["observation"]["timestep"], [0, 1, 2, 3])


def test_frequency_registry_matches_mystudy_for_known_sources():
    assert get_oxe_control_frequency_hz("fractal20220817_data") == 3.0
    assert get_oxe_control_frequency_hz("libero_spatial_no_noops") == 20.0
    with pytest.raises(ValueError, match="invalid control frequency"):
        get_oxe_control_frequency_hz("stanford_mask_vit_converted_externally_to_rlds")
