# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("scipy.spatial.transform")
pytest.importorskip("datasets")

import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import load_stats, write_stats, write_tasks
from lerobot.scripts.convert_absolute_eef_to_relative import (
    RAW_STATS_PATH,
    REPORT_PATH,
    convert_dataset,
    convert_poses,
    numeric_column,
)


def poses(position, matrices, gripper):
    return np.concatenate((position, matrices[:, :, 0], matrices[:, :, 1], gripper[:, None]), axis=1).astype(
        np.float32
    )


def test_base_frame_rotation_composition_and_gripper():
    # Noncommuting rotations distinguish base-frame deltas from local-frame
    # deltas, Euler subtraction and absolute rotation-vector subtraction.
    current = Rotation.from_euler("xyz", [[0.7, -0.4, 1.1], [0.1, 0.4, -0.3]]).as_matrix()
    step = Rotation.from_rotvec([[0.2, -0.1, 0.3], [-0.3, 0.2, 0.1]]).as_matrix()
    xyz = np.array([[0.3, -0.2, 0.5], [0.4, 0.3, 0.2]])
    offset = np.array([[0.01, -0.02, 0.03], [-0.02, 0.01, -0.03]])
    state = poses(xyz, current, np.array([-1, 0.5]))
    action = poses(xyz + offset, step @ current, np.array([1, -0.8]))
    delta, observation = convert_poses(action, state)
    assert delta.dtype == observation.dtype == np.float32
    np.testing.assert_allclose(delta[:, :3], offset, atol=5e-8)
    np.testing.assert_allclose(Rotation.from_rotvec(delta[:, 3:6]).as_matrix(), step, atol=1e-7)
    np.testing.assert_allclose(Rotation.from_rotvec(observation[:, 3:6]).as_matrix(), current, atol=1e-7)
    np.testing.assert_array_equal(delta[:, 6], action[:, 9])
    np.testing.assert_array_equal(observation[:, 6], state[:, 9])


def test_rotation_identity_small_angles_and_pi_reconstruct_targets():
    current = Rotation.from_rotvec([[0, 0, 0], [1e-9, -2e-9, 0], [0.3, -0.2, 0.4], [np.pi, 0, 0]])
    step = Rotation.from_rotvec([[0, 0, 0], [1e-9, 0, 0], [0, np.pi, 0], [0, 0, 1e-6]])
    xyz = np.zeros((4, 3))
    state = poses(xyz, current.as_matrix(), np.zeros(4))
    action = poses(xyz + 0.01, (step * current).as_matrix(), np.ones(4))
    delta, observation = convert_poses(action, state)
    recovered = Rotation.from_rotvec(delta[:, 3:6]) * Rotation.from_rotvec(observation[:, 3:6])
    np.testing.assert_allclose(recovered.as_matrix(), (step * current).as_matrix(), atol=3e-7)
    np.testing.assert_allclose(observation[:, :3] + delta[:, :3], action[:, :3], atol=1e-8)


@pytest.mark.parametrize("bad", ["nan", "zero_rotation", "parallel_rotation", "wrong_dimension"])
def test_reject_invalid_poses(bad):
    state = poses(np.zeros((1, 3)), np.eye(3)[None], np.zeros(1))
    action = state.copy()
    if bad == "nan":
        action[0, 0] = np.nan
    elif bad == "zero_rotation":
        action[0, 3:9] = 0
    elif bad == "parallel_rotation":
        action[0, 6:9] = action[0, 3:6]
    else:
        action = action[:, :7]
    with pytest.raises(ValueError):
        convert_poses(action, state)


@pytest.fixture
def source_dataset(tmp_path):
    root = tmp_path / "absolute"
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "videos/example.mp4").write_bytes(b"unchanged video bytes")
    current = Rotation.from_euler("xyz", [[0, 0, 0], [0.2, 0.1, 0.4], [0.4, -0.1, 0.2], [1, 0, -0.2]])
    step = Rotation.from_rotvec([[0.01, 0, 0], [0, 0.02, 0], [0, 0, -0.03], [0.01, 0.02, 0.03]])
    xyz = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [2, 3, 4], [3, 4, 5]])
    state = poses(xyz, current.as_matrix(), np.array([-1, 0.2, 0.5, 1]))
    action = poses(xyz + 0.01, (step * current).as_matrix(), np.array([0.2, 0.3, 0.8, -0.7]))
    features = {
        "action": {"dtype": "float32", "shape": [10], "names": None},
        "observation.state": {"dtype": "float32", "shape": [10], "names": None},
        **{
            key: {"dtype": "int64", "shape": [1], "names": None}
            for key in ("episode_index", "frame_index", "index", "task_index")
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v3.0",
        "features": features,
        "fps": 20,
        "total_frames": 4,
        "total_episodes": 2,
        "total_tasks": 1,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/labutopia_contract.json").write_text(
        json.dumps({"eef_frame": "panda_link8", "output_representation": "absolute"})
    )
    for ep in range(2):
        sl = slice(2 * ep, 2 * ep + 2)
        table = pa.table(
            {
                "action": pa.FixedSizeListArray.from_arrays(pa.array(action[sl].ravel()), 10),
                "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state[sl].ravel()), 10),
                "episode_index": [ep, ep],
                "frame_index": [0, 1],
                "index": [ep * 2, ep * 2 + 1],
                "task_index": [0, 0],
                "timestamp": pa.array([0, 0.05], type=pa.float32()),
            }
        ).replace_schema_metadata({b"huggingface": b"stale 10D schema"})
        pq.write_table(table, root / f"data/chunk-000/file-{ep:03d}.parquet")
    episodes = pa.table(
        {
            "episode_index": [0, 1],
            "length": [2, 2],
            "tasks": [["test task"], ["test task"]],
            "dataset_from_index": [0, 2],
            "dataset_to_index": [2, 4],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 1],
            "stats/action/mean": [[999.0] * 10] * 2,
            "stats/observation.state/mean": [[999.0] * 10] * 2,
        }
    )
    pq.write_table(episodes, root / "meta/episodes/chunk-000/file-000.parquet")
    write_tasks(pd.DataFrame({"task_index": [0]}, index=["test task"]), root)
    normalization = {
        name: {
            stat: np.full(10, i + 1, dtype=np.float32)
            for i, stat in enumerate(("min", "max", "mean", "std", "q01", "q99"))
        }
        for name in ("labutopia_absolute_eef", "labutopia_absolute_eef_state")
    }
    np.save(root / "meta/normalization_stats.npy", normalization)
    write_stats(
        {"observation.images.existing": {"mean": np.zeros((3, 1, 1)), "std": np.ones((3, 1, 1))}}, root
    )
    return root, action, state, normalization


def checksums(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_conversion_preserves_raw_values_and_rebuilds_metadata(source_dataset):
    root, action, state, normalization = source_dataset
    before = checksums(root)
    destination = convert_dataset(root)
    assert checksums(root) == before
    table = pa.concat_tables(
        [pq.read_table(path) for path in sorted((destination / "data").glob("*/*.parquet"))]
    )
    np.testing.assert_array_equal(numeric_column(table, "raw_action", (10,)), action)
    np.testing.assert_array_equal(numeric_column(table, "raw_state", (10,)), state)
    converted_action = numeric_column(table, "action", (7,))
    converted_state = numeric_column(table, "observation.state", (7,))
    # Different episode origins must not induce a spurious inter-episode delta.
    np.testing.assert_allclose(converted_action[:, :3], 0.01, atol=3e-7)
    info = json.loads((destination / "meta/info.json").read_text())
    for key in ("action", "observation.state"):
        assert info["features"][key]["shape"] == [7]
        assert table.schema.field(key).type == pa.list_(pa.float32(), 7)
    for key in ("raw_action", "raw_state"):
        assert info["features"][key]["shape"] == [10]
    stats = load_stats(destination)
    for key, values in (
        ("action", converted_action),
        ("observation.state", converted_state),
        ("raw_action", action),
        ("raw_state", state),
    ):
        np.testing.assert_allclose(stats[key]["mean"], values.astype(float).mean(0), atol=1e-12)
        np.testing.assert_allclose(stats[key]["std"], values.astype(float).std(0), atol=1e-12)
        np.testing.assert_allclose(stats[key]["q50"], np.quantile(values, 0.5, axis=0))
        np.testing.assert_array_equal(stats[key]["count"], [4])
    np.testing.assert_array_equal(stats["observation.images.existing"]["std"], np.ones((3, 1, 1)))
    exported = json.loads((destination / RAW_STATS_PATH).read_text())
    np.testing.assert_array_equal(
        exported["raw_action"]["mean"], normalization["labutopia_absolute_eef"]["mean"]
    )
    episodes = pq.read_table(destination / "meta/episodes/chunk-000/file-000.parquet")
    for ep in range(2):
        np.testing.assert_allclose(
            episodes["stats/action/mean"][ep].as_py(),
            converted_action[ep * 2 : ep * 2 + 2].astype(float).mean(0),
        )
        assert len(episodes["stats/raw_state/q99"][ep].as_py()) == 10
    assert (destination / "videos/example.mp4").read_bytes() == b"unchanged video bytes"
    assert table.schema.metadata is None
    report = json.loads((destination / REPORT_PATH).read_text())
    assert report["training_use_relative_actions"] is False
    contract = json.loads((destination / "meta/labutopia_contract.json").read_text())
    assert contract["output_representation"] == "relative_eef7_base_rotvec"
    meta = LeRobotDatasetMetadata("test/converted", root=destination)
    assert meta.info.features["action"]["shape"] == (7,)
    assert len(meta.episodes) == 2
    with pytest.raises(FileExistsError):
        convert_dataset(root)
    with pytest.raises(ValueError, match="already contains"):
        convert_dataset(destination)


@pytest.mark.parametrize("location", ["same", "nested", "parent"])
def test_reject_overlapping_output(source_dataset, location):
    root, *_ = source_dataset
    destination = {"same": root, "nested": root / "output", "parent": root.parent}[location]
    with pytest.raises(ValueError, match="outside"):
        convert_dataset(root, destination)


def test_invalid_episode_metadata_leaves_no_partial_output(source_dataset):
    root, *_ = source_dataset
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index("length"), "length", pa.array([3, 2]))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="inconsistent length"):
        convert_dataset(root)
    assert not root.with_name(f"{root.name}_relative_eef7").exists()
    assert not list(root.parent.glob(".*-converting-*"))


def test_null_vectors_are_rejected():
    table = pa.table({"action": pa.array([[1.0] * 9 + [None]], type=pa.list_(pa.float32(), 10))})
    with pytest.raises(ValueError, match="null"):
        numeric_column(table, "action", (10,))
