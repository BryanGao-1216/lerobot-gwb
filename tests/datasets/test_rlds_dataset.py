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

import io
import json
import pickle
import tarfile

import numpy as np
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.rlds_dataset import (
    ActionMemRLDSDataset,
    RLDSActionTokenCollator,
    _adapt_openx_tar_source_kwargs,
    _aggregate_weighted_stats,
    _attach_normalized_action_tokenizer_input,
    _filter_vqvla_action_chunk,
    _load_mixture_spec,
    _repeat_last_action_for_padded_targets,
    _resolve_rlds_source_format,
    _slice_action_chunk_with_last_frame,
    _validate_statistics,
)
from lerobot.datasets.rlds_webdataset import (
    iter_openx_tar_episodes,
    load_or_compute_openx_tar_statistics,
    resolve_openx_tar_paths,
    stack_openx_episode_steps,
)
from lerobot.utils.constants import (
    ACTION_TOKEN_DISTANCES,
    ACTION_TOKENIZER_INPUT,
)


class _NumpyTensorFlow:
    float32 = np.float32
    bool = np.bool_

    @staticmethod
    def cast(value, dtype):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def convert_to_tensor(value, dtype):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def where(condition, true_value, false_value):
        return np.where(condition, true_value, false_value)

    @staticmethod
    def equal(left, right):
        return np.equal(left, right)

    @staticmethod
    def clip_by_value(value, minimum, maximum):
        return np.clip(value, minimum, maximum)

    @staticmethod
    def zeros_like(value):
        return np.zeros_like(value)

    @staticmethod
    def shape(value):
        return np.asarray(value).shape

    @staticmethod
    def broadcast_to(value, shape):
        return np.broadcast_to(value, shape)

    @staticmethod
    def range(limit):
        return np.arange(limit)


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


def _write_openx_tar(path, payload):
    serialized = pickle.dumps(payload)
    member = tarfile.TarInfo("sample_000000000000.data.pickle")
    member.size = len(serialized)
    with tarfile.open(path, "w") as archive:
        archive.addfile(member, io.BytesIO(serialized))


def test_local_openx_tar_is_detected_and_streamed_without_extraction(tmp_path):
    dataset_dir = tmp_path / "austin_buds_dataset_converted_externally_to_rlds"
    dataset_dir.mkdir()
    tar_path = dataset_dir / "austin_buds_dataset_converted_externally_to_rlds_00000.tar"
    payload = {
        "steps": [
            {
                "action": np.array([1.0, 2.0], dtype=np.float32),
                "observation": {
                    "state": np.array([3.0], dtype=np.float32),
                    "image": {"bytes": b"first", "path": None},
                },
                "language_instruction": "pick",
            },
            {
                "action": np.array([4.0, 5.0], dtype=np.float32),
                "observation": {
                    "state": np.array([6.0], dtype=np.float32),
                    "image": {"bytes": b"second", "path": None},
                },
                "language_instruction": "pick",
            },
        ]
    }
    _write_openx_tar(tar_path, payload)

    paths = resolve_openx_tar_paths(tmp_path, dataset_dir.name)
    loaded = next(iter_openx_tar_episodes(paths))
    trajectory = stack_openx_episode_steps(loaded)

    assert paths == (tar_path,)
    assert np.array_equal(trajectory["action"], [[1.0, 2.0], [4.0, 5.0]])
    assert trajectory["observation"]["image"].tolist() == [b"first", b"second"]
    assert trajectory["language_instruction"].tolist() == ["pick", "pick"]
    assert list(dataset_dir.glob("*.data.pickle")) == []


def test_rlds_source_format_auto_prefers_tar_and_explicit_webdataset_requires_it(tmp_path):
    dataset_dir = tmp_path / "droid"
    dataset_dir.mkdir()
    (dataset_dir / "droid_00000.tar").touch()

    auto = DatasetConfig(repo_id="droid", root=str(tmp_path), rlds_storage_format="auto")
    assert _resolve_rlds_source_format(auto, "droid") == "webdataset"
    assert _resolve_rlds_source_format(auto, "missing") == "tfds"

    hybrid = DatasetConfig(repo_id="droid", root=str(tmp_path), rlds_storage_format="hybrid")
    assert _resolve_rlds_source_format(hybrid, "droid") == "webdataset"
    assert _resolve_rlds_source_format(hybrid, "missing") == "tfds"

    explicit = DatasetConfig(repo_id="missing", root=str(tmp_path), rlds_storage_format="webdataset")
    with pytest.raises(FileNotFoundError, match="no tar shards"):
        _resolve_rlds_source_format(explicit, "missing")


def test_hybrid_iterator_preserves_total_backend_weights():
    dataset = object.__new__(ActionMemRLDSDataset)
    dataset.seed = 7
    dataset.rank = 0
    dataset._hybrid_backend_weights = np.asarray([0.3, 0.7], dtype=np.float64)
    dataset._iter_weighted_webdatasets = lambda: iter(lambda: {"backend": "webdataset"}, None)
    dataset._iter_tfds_frames = lambda: iter(lambda: {"backend": "tfds"}, None)

    iterator = iter(dataset._iter_hybrid_frames())
    frames = (next(iterator) for _ in range(20_000))
    webdataset_fraction = sum(frame["backend"] == "webdataset" for frame in frames) / 20_000

    assert webdataset_fraction == pytest.approx(0.3, abs=0.015)


def test_rlds_overfit_iterator_caches_and_repeats_only_fixed_samples():
    dataset = object.__new__(ActionMemRLDSDataset)
    dataset._overfit_num_samples = 3
    dataset._overfit_samples = None
    dataset.rank = 0
    dataset.seed = 7
    consumed = []

    def source_frames():
        for value in range(10):
            consumed.append(value)
            yield {"value": value}

    dataset._iter_source_frames = source_frames
    dataset._to_lerobot_sample = lambda frame: {"value": frame["value"]}

    iterator = iter(dataset)
    values = [next(iterator)["value"] for _ in range(9)]

    assert consumed == [0, 1, 2]
    assert len(dataset._overfit_samples) == 3
    assert all(set(values[offset : offset + 3]) == {0, 1, 2} for offset in (0, 3, 6))


def test_bridge_tar_uses_openx_schema_for_action_images_and_state(tmp_path):
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge_00000.tar").touch()
    original_transform = object()
    kwargs = {
        "name": "bridge_orig",
        "standardize_fn": original_transform,
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
    }

    bridge_transform = object()
    adapted = _adapt_openx_tar_source_kwargs(
        kwargs,
        tmp_path,
        standardization_transforms={"bridge_oxe": bridge_transform},
    )

    assert adapted["standardize_fn"] is bridge_transform
    assert adapted["image_obs_keys"] == {"primary": "image", "secondary": None, "wrist": None}
    assert adapted["state_obs_keys"] == ["EEF_state", None, "gripper_state"]
    assert kwargs["image_obs_keys"]["primary"] == "image_0"


def test_local_openx_tar_statistics_are_computed_and_cached(tmp_path):
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    tar_path = dataset_dir / "toy_00000.tar"
    _write_openx_tar(
        tar_path,
        {
            "steps": [
                {
                    "action": np.array([1.0, 3.0], dtype=np.float32),
                    "observation": {"state": np.array([2.0], dtype=np.float32)},
                },
                {
                    "action": np.array([5.0, 7.0], dtype=np.float32),
                    "observation": {"state": np.array([6.0], dtype=np.float32)},
                },
            ]
        },
    )

    class _FakeTensorFlow:
        string = "string"

        @staticmethod
        def convert_to_tensor(value, dtype=None):
            return np.asarray(value, dtype=object if dtype == "string" else None)

    statistics = load_or_compute_openx_tar_statistics(
        paths=(tar_path,),
        tf=_FakeTensorFlow,
        restructure_fn=lambda trajectory: {
            "action": trajectory["action"],
            "observation": {"proprio": trajectory["observation"]["state"]},
        },
        hash_dependencies=("toy",),
    )

    assert statistics["num_trajectories"] == 1
    assert statistics["num_transitions"] == 2
    assert statistics["action"]["mean"] == [3.0, 5.0]
    assert statistics["proprio"]["mean"] == [4.0]
    assert len(list(dataset_dir.glob("dataset_statistics_*.json"))) == 1


def test_webdataset_image_decoder_accepts_predecoded_images():
    dataset = ActionMemRLDSDataset.__new__(ActionMemRLDSDataset)
    dataset.dataset_config = DatasetConfig(repo_id="toy", rlds_resize_size=(2, 3))

    image = dataset._decode_webdataset_image(np.full((4, 5, 3), 127, dtype=np.uint8))

    assert image.shape == (2, 3, 3)
    assert image.dtype == np.uint8


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


def test_weighted_action_stats_preserve_oxe_normalization_mask():
    statistics = [
        {
            "num_transitions": 10,
            "action": {
                "mean": np.array([0.0, 0.5]),
                "std": np.array([1.0, 0.5]),
                "min": np.array([-1.0, 0.0]),
                "max": np.array([1.0, 1.0]),
                "q01": np.array([-0.9, 0.0]),
                "q99": np.array([0.9, 1.0]),
                "mask": np.array([True, False]),
            },
        },
        {
            "num_transitions": 10,
            "action": {
                "mean": np.array([1.0, 0.25]),
                "std": np.array([2.0, 0.4]),
                "min": np.array([-2.0, 0.0]),
                "max": np.array([2.0, 1.0]),
                "q01": np.array([-1.8, 0.0]),
                "q99": np.array([1.8, 1.0]),
                "mask": np.array([True, False]),
            },
        },
    ]

    result = _aggregate_weighted_stats(
        statistics,
        weights=np.array([0.5, 0.5]),
        feature="action",
        size=2,
    )

    assert torch.equal(result["mask"], torch.tensor([True, False]))


def test_rlds_statistics_must_match_effect_tokenizer_action_dimension():
    statistics = {
        "action": {"mean": np.zeros(8)},
        "proprio": {"mean": np.zeros(7)},
    }

    with pytest.raises(ValueError, match="action tokenizer expects 7"):
        _validate_statistics("droid", statistics, action_dim=7, state_dim=32)


def test_action_tokenizer_input_matches_oxe_q01_q99_and_preserves_gripper():
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

    result = _attach_normalized_action_tokenizer_input(trajectory, statistics, _NumpyTensorFlow)

    assert np.array_equal(result["action"], raw_action)
    assert np.allclose(
        result[ACTION_TOKENIZER_INPUT],
        [[[-0.0, 0.0, -3.0], [1.0, 1.0, 4.0]]],
    )


def test_rlds_collator_encodes_only_normalized_action_tokenizer_input():
    class _RecordingEncoder:
        def __init__(self):
            self.actions = None

        def compute_code_distances(self, actions):
            self.actions = actions.clone()
            distances = torch.ones(actions.shape[0], 256, device=actions.device)
            distances[:, 17] = 0
            return distances

    collator = object.__new__(RLDSActionTokenCollator)
    collator.device = torch.device("cpu")
    collator.encoder = _RecordingEncoder()
    raw_action = torch.full((2, 3), 10.0)
    normalized_action = torch.full((2, 3), 0.25)

    batch = collator(
        [
            {
                "action": raw_action,
                ACTION_TOKENIZER_INPUT: normalized_action,
            }
        ]
    )

    assert torch.equal(collator.encoder.actions, normalized_action.unsqueeze(0))
    assert torch.equal(batch["action"], raw_action.unsqueeze(0))
    assert torch.equal(batch["action_token"], torch.tensor([17]))
    assert batch[ACTION_TOKEN_DISTANCES].shape == (1, 256)
    assert batch[ACTION_TOKEN_DISTANCES][0].argmin().item() == 17
    assert torch.equal(batch[ACTION_TOKENIZER_INPUT], normalized_action.unsqueeze(0))


def test_vqvla_chunk_filter_reads_shared_normalized_action_without_overwriting_raw_action():
    frame = {
        "action": np.full((2, 2), 10.0, dtype=np.float32),
        ACTION_TOKENIZER_INPUT: np.full((2, 2), 0.25, dtype=np.float32),
    }
    observed = {}

    def chunk_filter(candidate):
        observed["action"] = candidate["action"].copy()
        return True

    assert _filter_vqvla_action_chunk(frame, chunk_filter)
    assert np.all(observed["action"] == 0.25)
    assert np.all(frame["action"] == 10.0)


def test_tfds_tail_chunks_repeat_final_action_in_every_dimension():
    trajectory = {
        "action": np.array(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [0.0, 0.0]],
                [[2.0, 20.0], [3.0, 30.0], [0.0, 0.0], [0.0, 0.0]],
                [[3.0, 30.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    }

    result = _repeat_last_action_for_padded_targets(trajectory, horizon=4, tf=_NumpyTensorFlow)

    assert np.array_equal(
        result["action"],
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [3.0, 30.0]],
            [[2.0, 20.0], [3.0, 30.0], [3.0, 30.0], [3.0, 30.0]],
            [[3.0, 30.0], [3.0, 30.0], [3.0, 30.0], [3.0, 30.0]],
        ],
    )


def test_webdataset_tail_chunks_repeat_final_action_and_keep_last_start():
    action = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    penultimate = _slice_action_chunk_with_last_frame(action, start=1, horizon=4)
    final = _slice_action_chunk_with_last_frame(action, start=2, horizon=4)

    assert np.array_equal(penultimate[:, 0], [2.0, 3.0, 3.0, 3.0])
    assert np.array_equal(final[:, 0], [3.0, 3.0, 3.0, 3.0])


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
        ACTION_TOKENIZER_INPUT: np.full((16, 7), 0.25, dtype=np.float32),
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
    assert torch.all(sample[ACTION_TOKENIZER_INPUT] == 0.25)
    assert sample["observation.state"].shape == (10,)
    assert sample["observation.images.image"].shape == (3, 8, 8)
    assert sample["observation.images.image3_padding_mask"].item() is False
    assert sample["task"] == "pick up the cup"
    assert sample["dataset_name"] == "droid"
