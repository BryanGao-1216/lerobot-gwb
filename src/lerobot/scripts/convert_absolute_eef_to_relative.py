#!/usr/bin/env python

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

r"""Convert LabUtopia absolute flange poses to 7D base-frame delta actions.

    uv run python -m lerobot.scripts.convert_absolute_eef_to_relative \
        --root /path/to/level3_heat_200_absolute_flange

Input action and observation.state are float32, ordered as
[x, y, z, r00, r10, r20, r01, r11, r21, gripper]. The two rotation
columns describe the flange orientation in the robot base frame.

Output:
  raw_action / raw_state: the original 10D columns, unchanged;
  action: [p_action - p_state, rotvec(R_action @ R_state.T), action_gripper];
  observation.state: [p_state, rotvec(R_state), state_gripper].
Both new columns are float32 with shape (7,). Positions are metres, rotations
are axis-angle rotation vectors in radians, and grippers retain their original
absolute values/convention. Deltas use the action/state pair in the SAME row;
there is no temporal shift, cross-episode differencing, scaling or clipping.
For execution, p_target = p_current + delta_xyz and
R_target = Exp(delta_rotvec) @ R_current. For action chunks, execute each delta
against its corresponding current state, not one fixed chunk-start state.
Disable any further action-minus-state transform during training.

Default output is <root>_relative_eef7. The source is never overwritten and
existing outputs are refused. Conversion is staged before a final rename.
Videos are hard-linked when possible (shared storage), otherwise copied; use
--copy-videos for independent files. Video contents/timestamps are unchanged.

meta/stats.json and episode stats are recomputed from all numeric output
samples, with population std and exact q01/q10/q50/q90/q99. Old 10D absolute
stats cannot be transformed into 7D relative stats by slicing or subtraction.
The original normalization_stat(s).npy is retained and also exported as
meta/raw_normalization_stats.json, mapped to raw_action/raw_state. This export
preserves the original normalization parameters, including fixed rotation
ranges, separately from empirical stats.json. Existing image stats are kept;
missing image stats cannot be recovered from the NPY and are not invented.
No video decoding is needed. Memory usage is O(number of numeric samples).
Requires the dataset and scipy-dep extras.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError("Install dataset dependencies with: uv sync --locked --extra dataset") from exc

from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.utils import serialize_dict
from lerobot.utils.import_utils import require_package
from lerobot.utils.io_utils import write_json

ACTION = "action"
STATE = "observation.state"
RAW_ACTION = "raw_action"
RAW_STATE = "raw_state"
REPORT_PATH = "meta/eef_conversion.json"
RAW_STATS_PATH = "meta/raw_normalization_stats.json"
RAW_NAMES = ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"]
STATE_NAMES = ["x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z", "gripper"]
ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rotvec_x",
    "delta_rotvec_y",
    "delta_rotvec_z",
    "gripper",
]
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def rotation_matrices(poses: np.ndarray) -> np.ndarray:
    """Decode the two COLUMN vectors, rejecting invalid rotations before conversion."""
    if poses.ndim != 2 or poses.shape[1] != 10 or not np.isfinite(poses).all():
        raise ValueError("Expected finite poses with shape (N, 10).")
    first, second = poses[:, 3:6].astype(np.float64), poses[:, 6:9].astype(np.float64)
    first_norm, second_norm = np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)
    dot = np.sum(first * second, axis=1)
    valid = (abs(first_norm - 1) <= 1e-3) & (abs(second_norm - 1) <= 1e-3) & (abs(dot) <= 1e-3)
    if not valid.all():
        row = int(np.flatnonzero(~valid)[0])
        raise ValueError(f"Invalid rotation columns at row {row}: expected orthonormal column vectors.")
    # Remove only floating-point roundoff; preserve original bytes in raw_*.
    first /= first_norm[:, None]
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def convert_poses(action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 7D base-frame delta actions and 7D absolute observation states."""
    require_package("scipy", extra="scipy-dep")
    from scipy.spatial.transform import Rotation

    if action.shape != state.shape or len(action) == 0:
        raise ValueError("Action/state must have identical, nonempty shapes.")
    target, current = rotation_matrices(action), rotation_matrices(state)
    relative = target @ current.transpose(0, 2, 1)
    delta = np.concatenate(
        (
            action[:, :3].astype(np.float64) - state[:, :3],
            Rotation.from_matrix(relative).as_rotvec(),
            action[:, 9:10],
        ),
        axis=1,
    ).astype(np.float32)
    observation = np.concatenate(
        (state[:, :3], Rotation.from_matrix(current).as_rotvec(), state[:, 9:10]), axis=1
    ).astype(np.float32)
    return delta, observation


def numeric_column(table: pa.Table, key: str, shape: tuple[int, ...]) -> np.ndarray:
    column = table[key].combine_chunks()
    if column.null_count:
        raise ValueError(f"Null values in {key}.")
    if pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type):
        flat = column.flatten()
        if len(shape) != 1 or flat.null_count or not np.all(column.value_lengths().to_numpy() == shape[0]):
            raise ValueError(f"Invalid vector shape/null values in {key}; expected {shape}.")
        values = flat.to_numpy(zero_copy_only=False).reshape(len(column), shape[0])
    else:
        if shape != (1,):
            raise ValueError(f"Expected a vector column for {key} with shape {shape}.")
        values = column.to_numpy(zero_copy_only=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {key}.")
    return values


def feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = values.astype(np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)], dtype=np.int64),
        **{
            f"q{int(q * 100):02d}": result
            for q, result in zip(QUANTILES, np.quantile(values, QUANTILES, axis=0), strict=True)
        },
    }


def load_original_normalization(root: Path) -> tuple[Path, dict]:
    candidates = [root / "meta" / name for name in ("normalization_stats.npy", "normalization_stat.npy")]
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one meta/normalization_stats.npy or normalization_stat.npy.")
    path = candidates[0]
    # LabUtopia saves a trusted local dictionary as a NumPy object scalar.
    source = np.load(path, allow_pickle=True).item()
    if not isinstance(source, dict):
        raise ValueError("Normalization NPY must contain a dictionary.")
    mapped = {}
    for source_key, key in (
        ("labutopia_absolute_eef", RAW_ACTION),
        ("labutopia_absolute_eef_state", RAW_STATE),
    ):
        values = source.get(source_key)
        if not isinstance(values, dict) or not {"min", "max", "mean", "std", "q01", "q99"} <= values.keys():
            raise ValueError(f"Missing normalization parameters for {source_key}.")
        for name, value in values.items():
            array = np.asarray(value)
            expected_shape = (1,) if name == "count" else (10,)
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {source_key}/{name}: expected finite shape {expected_shape}.")
        mapped[key] = values
    return path, serialize_dict(mapped)


def read_numeric_data(root: Path, info: dict) -> tuple[list[Path], list[int], dict, dict]:
    features = {}
    for key, feature in info["features"].items():
        try:
            dtype = np.dtype(feature["dtype"])
        except TypeError:
            continue
        if dtype.kind in "biuf":
            features[key] = feature
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise ValueError("No LeRobot v3.0 data parquet files found.")
    collected, lengths = {key: [] for key in features}, []
    for path in tqdm(paths, desc="Reading numeric data"):
        table = pq.read_table(path, columns=list(features))
        for key, feature in features.items():
            values = numeric_column(table, key, tuple(feature["shape"]))
            if key in (ACTION, STATE) and values.dtype != np.float32:
                raise ValueError(f"Expected float32 storage for {key} in {path}.")
            collected[key].append(values)
        lengths.append(len(table))
    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    if sum(lengths) != info["total_frames"] or not np.array_equal(arrays["index"], np.arange(sum(lengths))):
        raise ValueError("Frame count/global indices are inconsistent with info.json or file order.")
    episodes, frames = arrays["episode_index"], arrays["frame_index"]
    ids, starts, counts = np.unique(episodes, return_index=True, return_counts=True)
    if not np.array_equal(ids, np.arange(info["total_episodes"])) or np.any(np.diff(episodes) < 0):
        raise ValueError("Episode indices must be contiguous, ordered, and consistent with info.json.")
    ranges = {
        int(ep): (int(start), int(start + count))
        for ep, start, count in zip(ids, starts, counts, strict=True)
    }
    for ep, (begin, end) in ranges.items():
        if not np.array_equal(frames[begin:end], np.arange(end - begin)):
            raise ValueError(f"Episode {ep} has missing/unordered frame indices.")
    return paths, lengths, arrays, ranges


def link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_episode_stats(stage: Path, arrays: dict, ranges: dict, info: dict, file_ranges: dict) -> None:
    per_episode = {
        ep: {key: feature_stats(values[begin:end]) for key, values in arrays.items()}
        for ep, (begin, end) in tqdm(ranges.items(), desc="Computing episode stats")
    }
    seen = set()
    for path in sorted((stage / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        ids = table["episode_index"].to_pylist()
        for row, ep in enumerate(ids):
            if ep in seen or ep not in ranges:
                raise ValueError(f"Unexpected/duplicate episode metadata: {ep}.")
            seen.add(ep)
            begin, end = ranges[ep]
            for key, expected in (
                ("length", end - begin),
                ("dataset_from_index", begin),
                ("dataset_to_index", end),
            ):
                if table[key][row].as_py() != expected:
                    raise ValueError(f"Episode {ep}: inconsistent {key}.")
            data_path = info["data_path"].format(
                chunk_index=table["data/chunk_index"][row].as_py(),
                file_index=table["data/file_index"][row].as_py(),
            )
            if data_path not in file_ranges or not (
                file_ranges[data_path][0] <= begin < end <= file_ranges[data_path][1]
            ):
                raise ValueError(f"Episode {ep}: data file reference does not contain its frames.")
        table = table.drop(
            [name for name in table.column_names if name.startswith(tuple(f"stats/{k}/" for k in arrays))]
        )
        for key in arrays:
            for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                dtype = pa.int64() if stat == "count" else pa.float64()
                column = pa.array([per_episode[ep][key][stat].tolist() for ep in ids], type=pa.list_(dtype))
                table = table.append_column(f"stats/{key}/{stat}", column)
        pq.write_table(table.replace_schema_metadata(None), path, compression="snappy")
    if seen != ranges.keys():
        raise ValueError("Episode metadata does not cover all data episodes.")


def convert_dataset(
    root: Path | str, output_dir: Path | str | None = None, copy_videos: bool = False
) -> Path:
    root = Path(root).expanduser().resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root.with_name(f"{root.name}_relative_eef7")
    )
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("Output must be outside the source dataset.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("This converter requires LeRobot v3.0 metadata.")
    if (root / REPORT_PATH).exists() or any(key in info["features"] for key in (RAW_ACTION, RAW_STATE)):
        raise ValueError("Dataset already contains raw/converted fields; use the original absolute dataset.")
    for key in (ACTION, STATE):
        feature = info["features"][key]
        if feature["shape"] != [10] or feature["dtype"] != "float32":
            raise ValueError(f"Expected a 10D float32 {key} feature.")
    npy_path, original_stats = load_original_normalization(root)
    paths, lengths, arrays, ranges = read_numeric_data(root, info)
    delta, observation = convert_poses(arrays[ACTION], arrays[STATE])
    arrays[RAW_ACTION], arrays[RAW_STATE] = arrays[ACTION], arrays[STATE]
    arrays[ACTION], arrays[STATE] = delta, observation
    for old, raw, names in ((ACTION, RAW_ACTION, ACTION_NAMES), (STATE, RAW_STATE, STATE_NAMES)):
        info["features"][raw] = {**deepcopy(info["features"][old]), "names": RAW_NAMES}
        info["features"][old] = {**info["features"][old], "shape": [7], "names": names}
    stats = load_stats(root) or {}
    stats.update({key: feature_stats(values) for key, values in arrays.items()})
    missing_image_stats = [
        key
        for key, feature in info["features"].items()
        if feature["dtype"] in ("image", "video") and not stats.get(key)
    ]
    report = {
        "converter": Path(__file__).name,
        "source_root": str(root),
        "source_normalization_file": npy_path.name,
        "source_normalization_sha256": hashlib.sha256(npy_path.read_bytes()).hexdigest(),
        "raw_fields": {ACTION: RAW_ACTION, STATE: RAW_STATE},
        "input_order": RAW_NAMES,
        "action_order": ACTION_NAMES,
        "state_order": STATE_NAMES,
        "coordinate_frame": "robot base",
        "position_delta": "p(raw_action[t]) - p(raw_state[t])",
        "rotation_delta": "rotvec(R(raw_action[t]) @ R(raw_state[t]).T)",
        "position_unit": "metres",
        "rotation_unit": "radians",
        "rotation_representation": "axis-angle rotation vector, principal angle <= pi",
        "gripper": "absolute source values preserved, not differenced",
        "state": "absolute measured pose, not a delta",
        "temporal_reference": "paired action/state in the same row; per-step deltas, not chunk-start offsets",
        "training_use_relative_actions": False,
        "total_frames": info["total_frames"],
        "statistics": "empirical full-dataset and per-episode numeric stats; population std; exact quantiles",
        "original_normalization_export": RAW_STATS_PATH,
        "missing_image_stats": missing_image_stats,
    }
    contract_path = root / "meta/labutopia_contract.json"
    if contract_path.exists():
        report["source_contract"] = json.loads(contract_path.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-converting-", dir=destination.parent))
    try:
        for entry in root.iterdir():
            if entry.name in {"data", ".cache", ".git", ".DS_Store"}:
                continue
            if entry.is_dir():
                copier = link_or_copy if entry.name == "videos" and not copy_videos else shutil.copy2
                shutil.copytree(entry, stage / entry.name, copy_function=copier)
            else:
                shutil.copy2(entry, stage / entry.name)
        start, file_ranges = 0, {}
        for path, length in tqdm(list(zip(paths, lengths, strict=True)), desc="Writing data"):
            table = pq.read_table(path)
            names = [{ACTION: RAW_ACTION, STATE: RAW_STATE}.get(key, key) for key in table.column_names]
            if len(names) != len(set(names)):
                raise ValueError(f"Raw column name collision in {path}.")
            table = table.rename_columns(names)
            for key in (ACTION, STATE):
                values = arrays[key][start : start + length]
                table = table.append_column(
                    key, pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 7)
                )
            relative_path = path.relative_to(root)
            target = stage / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table.replace_schema_metadata(None), target, compression="snappy")
            file_ranges[relative_path.as_posix()] = (start, start + length)
            start += length
        write_episode_stats(stage, arrays, ranges, info, file_ranges)
        write_stats(stats, stage)
        write_json(info, stage / "meta/info.json")
        write_json(original_stats, stage / RAW_STATS_PATH)
        write_json(report, stage / REPORT_PATH)
        if contract_path.exists():
            contract = {
                **report["source_contract"],
                "output_representation": "relative_eef7_base_rotvec",
                "eef_pose_convention": "xyz_rotvec_robot_base",
                "state": "measured absolute flange xyz, rotation vector and openness at t",
                "action": "base-frame translation/rotation delta relative to measured state at t; absolute openness",
                "raw_action": "original 10D absolute achieved flange pose and openness",
                "raw_state": "original 10D absolute measured flange pose and openness",
            }
            write_json(contract, stage / "meta/labutopia_contract.json")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Destination appeared during conversion: {destination}.")
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    if missing_image_stats:
        logging.warning(
            "NPY has no image statistics for %s; use image identity/ImageNet normalization or compute them separately.",
            missing_image_stats,
        )
    logging.info("Converted %d frames -> %s", info["total_frames"], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="Original absolute 10D LeRobot v3.0 dataset")
    parser.add_argument("--output-dir", type=Path, help="Default: <root>_relative_eef7")
    parser.add_argument(
        "--copy-videos", action="store_true", help="Copy videos instead of hard-linking where possible"
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    convert_dataset(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
