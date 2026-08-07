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

import json

import numpy as np
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.rlds_dataset import (
    _ACTION_VQVAE_INPUT,
    ActionMemRLDSDataset,
    RLDSActionTokenCollator,
    _aggregate_weighted_stats,
    _attach_normalized_action_vqvae_input,
    _load_mixture_spec,
    _make_actionmem_standardizer,
    _validate_statistics,
    _wrap_standardizers,
)


class _NumpyTensorFlow:
    float32 = np.float32
    bool = np.bool_
    math = type("Math", (), {"floormod": staticmethod(np.mod)})

    @staticmethod
    def cast(value, dtype):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def convert_to_tensor(value, dtype):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def concat(values, axis):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def where(condition, true_value, false_value):
        return np.where(condition, true_value, false_value)

    @staticmethod
    def ones_like(value):
        return np.ones_like(value)

    @staticmethod
    def equal(left, right):
        return np.equal(left, right)

    @staticmethod
    def clip_by_value(value, minimum, maximum):
        return np.clip(value, minimum, maximum)

    @staticmethod
    def zeros_like(value):
        return np.zeros_like(value)


def test_load_explicit_rlds_mixture(tmp_path):
    mixture_path = tmp_path / "mixture.json"
    mixture_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "droid", "weight": 2.0},
                    {"name": "rl_bench", "weight": 1.0},
                ]
            }
        )
    )
    config = DatasetConfig(repo_id="unused", rlds_mixture_path=str(mixture_path))

    assert _load_mixture_spec(config, {}) == [("droid", 2.0), ("rl_bench", 1.0)]


def test_load_named_or_single_rlds_mixture():
    named = {"research_mix": [("droid", 1.0), ("rl_bench", 0.5)]}

    named_config = DatasetConfig(repo_id="research_mix")
    single_config = DatasetConfig(repo_id="droid")

    assert _load_mixture_spec(named_config, named) == named["research_mix"]
    assert _load_mixture_spec(single_config, named) == [("droid", 1.0)]


def test_rlds_mixture_rejects_duplicate_sources():
    config = DatasetConfig(repo_id="duplicate_mix")
    named = {"duplicate_mix": [("droid", 1.0), ("droid", 0.5)]}

    with pytest.raises(ValueError, match="duplicate dataset"):
        _load_mixture_spec(config, named)


def test_weighted_stats_follow_effective_sampling_distribution_and_pad_state():
    statistics = [
        {
            "num_transitions": 10,
            "proprio": {
                "mean": np.array([0.0, 2.0]),
                "std": np.array([1.0, 2.0]),
                "min": np.array([-1.0, 0.0]),
                "max": np.array([1.0, 4.0]),
                "q01": np.array([-0.9, 0.1]),
                "q99": np.array([0.9, 3.9]),
            },
        },
        {
            "num_transitions": 30,
            "proprio": {
                "mean": np.array([4.0]),
                "std": np.array([3.0]),
                "min": np.array([1.0]),
                "max": np.array([7.0]),
                "q01": np.array([1.1]),
                "q99": np.array([6.9]),
            },
        },
    ]

    result = _aggregate_weighted_stats(
        statistics,
        weights=np.array([0.25, 0.75]),
        feature="proprio",
        size=3,
    )

    assert torch.allclose(result["mean"], torch.tensor([3.0, 0.5, 0.0]))
    expected_variance = torch.tensor([10.0, 1.75, 0.0])
    assert torch.allclose(result["std"].square(), expected_variance)
    assert result["count"].item() == 40


def test_actionmem_transform_rejects_unverified_dataset():
    kwargs = [{"name": "bridge_orig", "standardize_fn": lambda trajectory: trajectory}]

    with pytest.raises(ValueError, match="only verified for DROID, RLBench, and LIBERO"):
        _wrap_standardizers(kwargs, "actionmem", object())


def test_droid_standardizer_matches_vqvae_relative_eef_extraction():
    raw_action = np.array([[2.0, 3.0, 4.0, 3.2, -3.2, 0.2, 0.25]], dtype=np.float32)
    state = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.1]], dtype=np.float32)
    trajectory = {
        "action": raw_action,
        "observation": {"cartesian_position": state},
    }

    standardizer = _make_actionmem_standardizer(lambda value: value, "droid", _NumpyTensorFlow)
    action = standardizer(trajectory)["action"]

    expected_rotation = (raw_action[:, 3:6] - state[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    expected = np.concatenate(
        [raw_action[:, :3] - state[:, :3], expected_rotation, np.array([[-1.0]])], axis=-1
    )
    assert np.allclose(action, expected)


def test_rlbench_standardizer_matches_vqvae_relative_eef_extraction():
    raw_action = np.array([[2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.0]], dtype=np.float32)
    state = np.array([[1.0, 1.0, 1.0, 0.0, 0.1, 0.2, 9.0]], dtype=np.float32)
    trajectory = {"action": raw_action, "observation": {"state": state}}

    standardizer = _make_actionmem_standardizer(lambda value: value, "rl_bench", _NumpyTensorFlow)
    action = standardizer(trajectory)["action"]

    assert np.allclose(action, [[1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 1.0]])


def test_rlds_statistics_must_match_vqvae_action_dimension():
    statistics = {
        "action": {"mean": np.zeros(8)},
        "proprio": {"mean": np.zeros(7)},
    }

    with pytest.raises(ValueError, match="VQ-VAE expects 7"):
        _validate_statistics("droid", statistics, action_dim=7, state_dim=32)


def test_vqvae_action_input_uses_q01_q99_while_flow_action_stays_raw():
    raw_action = np.array(
        [[[0.0, 2.0, -3.0], [1.0, 3.0, 4.0]]],
        dtype=np.float32,
    )
    trajectory = {"action": raw_action.copy()}
    statistics = {
        "action": {
            "q01": np.array([-1.0, 1.0, -1.0], dtype=np.float32),
            "q99": np.array([1.0, 3.0, 1.0], dtype=np.float32),
            "min": np.array([-2.0, 0.0, -1.0], dtype=np.float32),
            "max": np.array([2.0, 4.0, 1.0], dtype=np.float32),
            "mask": np.array([True, True, False]),
        }
    }

    result = _attach_normalized_action_vqvae_input(trajectory, statistics, _NumpyTensorFlow)

    assert np.array_equal(result["action"], raw_action)
    assert np.allclose(
        result[_ACTION_VQVAE_INPUT],
        [[[-0.0, 0.0, -3.0], [1.0, 1.0, 4.0]]],
    )


def test_rlds_collator_encodes_only_normalized_vqvae_action_input():
    class _RecordingEncoder:
        def __init__(self):
            self.actions = None

        def __call__(self, actions):
            self.actions = actions.clone()
            return torch.tensor([17], dtype=torch.long, device=actions.device)

    collator = object.__new__(RLDSActionTokenCollator)
    collator.device = torch.device("cpu")
    collator.encoder = _RecordingEncoder()
    raw_action = torch.full((2, 3), 10.0)
    normalized_action = torch.full((2, 3), 0.25)

    batch = collator(
        [
            {
                "action": raw_action,
                _ACTION_VQVAE_INPUT: normalized_action,
            }
        ]
    )

    assert torch.equal(collator.encoder.actions, normalized_action.unsqueeze(0))
    assert torch.equal(batch["action"], raw_action.unsqueeze(0))
    assert torch.equal(batch["action_token"], torch.tensor([17]))
    assert _ACTION_VQVAE_INPUT not in batch


def test_rlds_frame_is_converted_to_lerobot_actionmem_sample():
    dataset = object.__new__(ActionMemRLDSDataset)
    dataset.action_horizon = 16
    dataset.action_dim = 7
    dataset.state_dim = 10
    dataset.dataset_config = DatasetConfig(
        repo_id="test", rlds_camera_views=("primary", "wrist"), rlds_resize_size=(8, 8)
    )
    frame = {
        "action": np.zeros((16, 7), dtype=np.float32),
        _ACTION_VQVAE_INPUT: np.full((16, 7), 0.25, dtype=np.float32),
        "dataset_name": b"droid",
        "task": {"language_instruction": b"pick up the cup"},
        "observation": {
            "proprio": np.ones((1, 7), dtype=np.float32),
            "image_primary": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "image_wrist": np.zeros((1, 8, 8, 3), dtype=np.uint8),
            "pad_mask_dict": {
                "image_primary": np.array([True]),
                "image_wrist": np.array([False]),
            },
        },
    }

    sample = dataset._to_lerobot_sample(frame)

    assert sample["action"].shape == (16, 7)
    assert torch.all(sample[_ACTION_VQVAE_INPUT] == 0.25)
    assert sample["observation.state"].shape == (10,)
    assert sample["observation.images.image"].shape == (3, 8, 8)
    assert sample["observation.images.image3_padding_mask"].item() is False
    assert sample["task"] == "pick up the cup"
    assert sample["dataset_name"] == "droid"
