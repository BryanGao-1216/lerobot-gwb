#!/usr/bin/env python
r"""Convert bimanual Piper joint actions to LIBERO-scaled flange delta actions.

Example (use the environment with this checkout and dataset/scipy-dep installed)::

    python -m lerobot.scripts.convert_piper_joint_to_ee_delta \
        --root ../Datasets/boil_0613 \
        --urdf ../Datasets/agx_arm_urdf/piper/urdf/piper_description.urdf

Default: save beside the source as <root>_ee_delta. Use --output-dir to choose
another destination, or --in-place to replace the source after conversion while
keeping the original directory as <root>_old. Existing outputs/backups are refused.

Input order: [left joint1..joint6, gripper, right joint1..joint6, gripper].
Input joint angles are radians. Output per arm is [dx, dy, dz, drx, dry, drz, gripper]:
  xyz = (p(action[t]) - p(observation.state[t])) / 0.05 metres;
  rotation = rotvec(R(action[t]) @ R(observation.state[t]).T) / 0.5 radians;
  gripper = -1 (open) if the raw absolute target is > 0.01, otherwise +1 (closed).
This default signed gripper encoding follows the LIBERO control interface. For
LIBERO datasets stored as 1=open/0=closed, use --gripper-encoding zero-one. The
same 0.01 threshold is applied before either encoding; equality means closed.
The six pose components are dimensionless: 1 means 0.05 m or 0.5 rad, respectively.
The endpoint is link6 (flange), expressed in that arm's base frame. Pose values
are scaled without clipping. The dataset FPS and action time span are preserved;
the output represents per-step commands, not velocities. State stays in joint
space with its original continuous gripper readings. Input must be the original
joint dataset, not an earlier converted output. Do not apply another
action-minus-state transform in training.

Isolated joint values outside URDF limits plus a 5-degree margin (or non-finite
values) are replaced by the mean of their immediate good neighbours in the same
episode, independently in action and observation.state. Boundary/consecutive bad
samples raise an error. State otherwise stays in joint space; velocity/effort stay
as recorded. Both episode and global statistics for the changed fields are rebuilt.
Quantiles are computed directly from all samples, including exact zero-heavy data.
Task text has underscores replaced with spaces in both the task table and episode
summaries. Identical resulting tasks are merged and frame task indices are remapped.

Only numeric columns are loaded for computation (O(number of frames) memory).
FK is vectorized in batches, and each data parquet is rewritten once. No video is
decoded/re-encoded: videos are hard-linked when possible, otherwise copied. Linked
videos share storage with the source/backup; use --copy-videos for independent files.
LeRobot supplies metadata/statistics I/O; batched NumPy/SciPy FK avoids a per-frame
robot solver and requires neither ROS, mesh loading, xacro nor placo.
"""

import argparse
import hashlib
import logging
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError("Install dataset dependencies with: uv sync --locked --extra dataset") from exc

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES
from lerobot.datasets.io_utils import load_info, load_stats, load_tasks, write_info, write_stats, write_tasks
from lerobot.utils.import_utils import require_package
from lerobot.utils.io_utils import write_json

ACTION = "action"
STATE = "observation.state"
TASK = "task_index"
REPORT = "meta/action_conversion.json"
JOINTS = [f"joint{i}" for i in range(1, 7)]
JOINT_COLUMNS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
COMPONENTS = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
POSITION_SCALE_M = 0.05
ROTATION_SCALE_RAD = 0.5
GRIPPER_OPEN_THRESHOLD = 0.01
GRIPPER_ENCODINGS = {"signed": (-1.0, 1.0), "zero-one": (1.0, 0.0)}  # (open, closed)


class PiperFK:
    """Evaluate the URDF base_link -> link6 chain for an entire batch."""

    def __init__(self, urdf: Path):
        require_package("scipy", extra="scipy-dep")
        from scipy.spatial.transform import Rotation

        self.rotation = Rotation
        robot = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
        chain, link, visited = [], "link6", set()
        while link != "base_link":
            if link in visited or link not in by_child:
                raise ValueError("URDF must contain a connected base_link -> link6 chain.")
            visited.add(link)
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        chain.reverse()
        if [j.get("name") for j in chain] != JOINTS:
            raise ValueError(f"Expected the Piper chain {JOINTS}, got {[j.get('name') for j in chain]}.")
        self.origins, self.axes, limits = [], [], []
        for joint in chain:
            if joint.get("type") != "revolute":
                raise ValueError(f"Expected a revolute joint: {joint.get('name')}")
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            if not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError(f"Invalid joint axis: {joint.get('name')}")
            self.origins.append(transform)
            self.axes.append(axis)
            limit = joint.find("limit")
            limits.append([float(limit.get("lower")), float(limit.get("upper"))])
        self.limits = np.asarray(limits)

    def __call__(self, joints: np.ndarray) -> np.ndarray:
        pose = np.broadcast_to(np.eye(4), (len(joints), 4, 4)).copy()
        turn = np.broadcast_to(np.eye(4), pose.shape).copy()
        for i, (origin, axis) in enumerate(zip(self.origins, self.axes, strict=True)):
            turn[:, :3, :3] = self.rotation.from_rotvec(joints[:, i, None] * axis).as_matrix()
            pose = pose @ origin @ turn
        return pose


def vector_column(table: pa.Table, key: str) -> np.ndarray:
    column = table[key].combine_chunks()
    if column.null_count or column.flatten().null_count:
        raise ValueError(f"Null values in {key}.")
    if not np.all(column.value_lengths().to_numpy() == 14):
        raise ValueError(f"Expected 14 values per row in {key}.")
    return column.flatten().to_numpy(zero_copy_only=False).reshape(-1, 14).astype(np.float32, copy=True)


def repair_spikes(values, episodes, frames, limits, margin, feature):
    """Repair only isolated, out-of-range joint components; never cross episodes."""
    joints = values[:, JOINT_COLUMNS]
    bounds = np.tile(limits, (2, 1))
    bad = ~np.isfinite(joints) | (joints < bounds[:, 0] - margin) | (joints > bounds[:, 1] + margin)
    repairs = []
    for row, joint in zip(*np.nonzero(bad), strict=True):
        if (
            row == 0
            or row == len(values) - 1
            or episodes[row - 1] != episodes[row]
            or episodes[row + 1] != episodes[row]
            or bad[row - 1, joint]
            or bad[row + 1, joint]
        ):
            raise ValueError(
                f"Cannot average {feature}, episode {episodes[row]}, frame {frames[row]}, "
                f"joint column {JOINT_COLUMNS[joint]}: two good same-episode neighbours are required."
            )
        col = int(JOINT_COLUMNS[joint])
        old = float(values[row, col])
        values[row, col] = (float(values[row - 1, col]) + float(values[row + 1, col])) / 2
        repairs.append(
            {
                "feature": feature,
                "episode_index": int(episodes[row]),
                "frame_index": int(frames[row]),
                "column": col,
                "before": old if np.isfinite(old) else str(old),
                "after": float(values[row, col]),
            }
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite gripper values in {feature}; joint repair cannot fix them.")
    logging.info("Repaired %d joint values in %s", len(repairs), feature)
    return repairs


def delta_actions(state, action, fk, batch_size, gripper_encoding="signed"):
    """Return both arms' scaled pose deltas and binary LIBERO gripper commands."""
    open_value, closed_value = GRIPPER_ENCODINGS[gripper_encoding]
    result = np.empty_like(action)
    for start in tqdm(range(0, len(action), batch_size), desc="Batched flange FK"):
        stop = min(start + batch_size, len(action))
        for offset in (0, 7):
            current = fk(state[start:stop, offset : offset + 6])
            target = fk(action[start:stop, offset : offset + 6])
            result[start:stop, offset : offset + 3] = (
                target[:, :3, 3] - current[:, :3, 3]
            ) / POSITION_SCALE_M
            relative = target[:, :3, :3] @ current[:, :3, :3].transpose(0, 2, 1)
            result[start:stop, offset + 3 : offset + 6] = (
                fk.rotation.from_matrix(relative).as_rotvec() / ROTATION_SCALE_RAD
            )
            result[start:stop, offset + 6] = np.where(
                action[start:stop, offset + 6] > np.float32(GRIPPER_OPEN_THRESHOLD), open_value, closed_value
            )
    if not np.isfinite(result).all():
        raise ValueError("FK produced non-finite actions.")
    return result


def replace_numeric(table, key, values):
    # Preserve source Arrow types without Python row loops.
    array = pa.array(values.ravel())
    if values.ndim == 2:
        array = pa.FixedSizeListArray.from_arrays(array, values.shape[1])
    return table.set_column(table.schema.get_field_index(key), key, array.cast(table.schema.field(key).type))


def feature_stats(values):
    values = values.astype(np.float64)
    if values.ndim == 1:
        values = values[:, None]
    # Histogram interpolation smears a point mass at zero across its bin. All
    # numeric samples are already in memory, so use exact global/episode quantiles.
    quantiles = np.quantile(values, DEFAULT_QUANTILES, axis=0)
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)]),
        **{f"q{int(q * 100):02d}": value for q, value in zip(DEFAULT_QUANTILES, quantiles, strict=True)},
    }


def normalize_tasks(root):
    """Replace underscores literally; merge only exact resulting text matches."""
    tasks = load_tasks(root).sort_values(TASK)
    if not np.array_equal(tasks[TASK].to_numpy(), np.arange(len(tasks))):
        raise ValueError("Task table must have unique, contiguous task indices starting at zero.")
    task_to_index, first_rows, lookup = {}, [], []
    for row, task in enumerate(tasks.index):
        if not isinstance(task, str):
            raise ValueError("Task descriptions must be strings.")
        normalized = task.replace("_", " ")
        if normalized not in task_to_index:
            task_to_index[normalized] = len(first_rows)
            first_rows.append(row)
        lookup.append(task_to_index[normalized])
    normalized_tasks = tasks.iloc[first_rows].copy()
    normalized_tasks.index = normalized_tasks.index.str.replace("_", " ", regex=False)
    normalized_tasks[TASK] = np.arange(len(normalized_tasks), dtype=np.int64)
    logging.info("Normalized task text: %d tasks -> %d unique tasks", len(tasks), len(normalized_tasks))
    return normalized_tasks, np.asarray(lookup, dtype=np.int64)


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_output(root, stage, paths, lengths, info, arrays, ranges, copy_videos, report):
    shutil.copytree(root / "meta", stage / "meta")
    for entry in root.iterdir():
        if entry.name in {"meta", "data", ".cache", ".git"}:
            continue
        if entry.is_dir():
            copier = link_or_copy if entry.name == "videos" and not copy_videos else shutil.copy2
            shutil.copytree(entry, stage / entry.name, copy_function=copier)
        else:
            shutil.copy2(entry, stage / entry.name)

    start = 0
    for path, length in tqdm(list(zip(paths, lengths, strict=True)), desc="Writing data"):
        table = pq.read_table(path)
        for key, values in arrays.items():
            table = replace_numeric(table, key, values[start : start + length])
        destination = stage / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="snappy")
        if pq.read_metadata(destination).num_rows != length:
            raise ValueError(f"Written frame count differs in {destination}")
        start += length

    per_episode = {
        ep: {key: feature_stats(values[begin:end]) for key, values in arrays.items()}
        for ep, (begin, end) in tqdm(ranges.items(), desc="Episode statistics")
    }
    seen = set()
    for path in sorted((stage / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        ids = table["episode_index"].to_pylist()
        for row, ep in enumerate(ids):
            if ep in seen or ep not in ranges:
                raise ValueError(f"Unexpected/duplicate episode metadata: {ep}")
            seen.add(ep)
            begin, end = ranges[ep]
            for key, expected in [
                ("length", end - begin),
                ("dataset_from_index", begin),
                ("dataset_to_index", end),
            ]:
                if table[key][row].as_py() != expected:
                    raise ValueError(f"Episode {ep}: inconsistent {key}.")
        if "tasks" in table.column_names:
            tasks = [
                list(dict.fromkeys(task.replace("_", " ") for task in episode_tasks))
                for episode_tasks in table["tasks"].to_pylist()
            ]
            table = table.set_column(
                table.schema.get_field_index("tasks"),
                "tasks",
                pa.array(tasks, type=table.schema.field("tasks").type),
            )
        table = table.drop(
            [c for c in table.column_names if c.startswith(tuple(f"stats/{k}/" for k in arrays))]
        )
        for key in arrays:
            for stat in per_episode[ids[0]][key]:
                column = pa.array([per_episode[ep][key][stat].tolist() for ep in ids])
                table = table.append_column(f"stats/{key}/{stat}", column)
        # These columns changed, so discard stale pandas/HF schema annotations.
        pq.write_table(table.replace_schema_metadata(None), path, compression="snappy")
    if seen != ranges.keys():
        raise ValueError("Episode metadata does not cover all data episodes.")

    stats = load_stats(root) or {}
    stats.update({key: feature_stats(values) for key, values in arrays.items()})
    write_stats(stats, stage)
    write_info(info, stage)
    write_json(report, stage / REPORT)


def convert_dataset(
    root,
    urdf,
    output_dir=None,
    in_place=False,
    batch_size=16384,
    joint_limit_margin_deg=5.0,
    copy_videos=False,
    gripper_encoding="signed",
):
    started = time.perf_counter()
    root, urdf = Path(root).expanduser().resolve(), Path(urdf).expanduser().resolve()
    if batch_size < 1 or not np.isfinite(joint_limit_margin_deg) or joint_limit_margin_deg < 0:
        raise ValueError("batch_size must be positive and joint_limit_margin_deg finite and non-negative.")
    if in_place and output_dir is not None:
        raise ValueError("Choose either --in-place or --output-dir.")
    if gripper_encoding not in GRIPPER_ENCODINGS:
        raise ValueError(f"Unknown gripper encoding: {gripper_encoding}")
    destination = (
        root
        if in_place
        else (
            Path(output_dir).expanduser().resolve() if output_dir else root.with_name(f"{root.name}_ee_delta")
        )
    )
    backup = root.with_name(f"{root.name}_old")
    if not in_place and (destination == root or root in destination.parents or destination in root.parents):
        raise ValueError("Output must be outside the source; use --in-place for replacement.")
    protected = backup if in_place else destination
    if protected.exists() or protected.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {protected}")
    if (root / REPORT).exists():
        raise ValueError(
            "This dataset already contains converted delta actions. "
            "Use the original joint dataset (or its _old backup) as --root."
        )
    info = load_info(root)
    if info.codebase_version != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.codebase_version}")
    for key in (STATE, ACTION):
        feature = info.features[key]
        if tuple(feature["shape"]) != (14,) or feature["dtype"] != "float32":
            raise ValueError(f"Expected a float32, 14-dimensional {key} feature.")

    fk = PiperFK(urdf)
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise ValueError("No V3 data parquet files found.")
    tasks, task_lookup = normalize_tasks(root)
    if len(task_lookup) != info.total_tasks:
        raise ValueError("Task table count differs from info.json.")
    info.total_tasks = len(tasks)
    columns = [STATE, ACTION, TASK, "episode_index", "frame_index", "index"]
    tables = [pq.read_table(path, columns=columns) for path in paths]
    lengths = [len(table) for table in tables]
    numeric = pa.concat_tables(tables)
    state, action = vector_column(numeric, STATE), vector_column(numeric, ACTION)
    task_indices = numeric[TASK].to_numpy()
    if (
        not np.issubdtype(task_indices.dtype, np.integer)
        or np.any(task_indices < 0)
        or np.any(task_indices >= len(task_lookup))
    ):
        raise ValueError("Data contains invalid task indices.")
    task_indices = task_lookup[task_indices]
    episodes = numeric["episode_index"].to_numpy()
    frames = numeric["frame_index"].to_numpy()
    if len(numeric) != info.total_frames or not np.array_equal(
        numeric["index"].to_numpy(), np.arange(len(numeric))
    ):
        raise ValueError("Data must contain all frames, ordered by contiguous global index.")
    ids, starts, counts = np.unique(episodes, return_index=True, return_counts=True)
    if len(ids) != info.total_episodes or np.any(np.diff(episodes) < 0):
        raise ValueError("Episode count/order is inconsistent with info.json.")
    ranges = {
        int(ep): (int(start), int(start + count))
        for ep, start, count in zip(ids, starts, counts, strict=True)
    }
    for ep, (begin, end) in ranges.items():
        if not np.array_equal(frames[begin:end], np.arange(end - begin)):
            raise ValueError(f"Episode {ep} has missing or unordered frame indices.")
    del numeric, tables
    repairs = []
    for key, values in [(STATE, state), (ACTION, action)]:
        repairs.extend(
            repair_spikes(values, episodes, frames, fk.limits, np.deg2rad(joint_limit_margin_deg), key)
        )
    converted = delta_actions(state, action, fk, batch_size, gripper_encoding)
    del action
    info.features[ACTION] = {
        **info.features[ACTION],
        "names": [f"{side}_{component}" for side in ("left", "right") for component in COMPONENTS],
    }
    open_value, closed_value = GRIPPER_ENCODINGS[gripper_encoding]
    report = {
        "converter": Path(__file__).name,
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "joint_names": JOINTS,
        "joint_units": "radians",
        "target_frame": "link6",
        "coordinate_frame": "per-arm base_link",
        "action_representation": "libero_scaled_delta",
        "position_delta": f"(p(action[t]) - p(observation.state[t])) / {POSITION_SCALE_M}; dimensionless",
        "rotation_delta": (
            f"rotvec(R(action[t]) @ R(observation.state[t]).T) / {ROTATION_SCALE_RAD}; dimensionless"
        ),
        "position_scale_m": POSITION_SCALE_M,
        "rotation_scale_rad": ROTATION_SCALE_RAD,
        "pose_clipping": False,
        "fps": info.fps,
        "gripper": (
            f"raw absolute action value > {GRIPPER_OPEN_THRESHOLD}: {open_value} (open); "
            f"otherwise {closed_value} (closed)"
        ),
        "gripper_encoding": gripper_encoding,
        "gripper_open_threshold": GRIPPER_OPEN_THRESHOLD,
        "gripper_open_value": open_value,
        "gripper_closed_value": closed_value,
        "observation_state": "joint positions with spikes repaired; continuous gripper readings preserved",
        "joint_limit_margin_deg": joint_limit_margin_deg,
        "repairs": repairs,
        "task_normalization": "Replace underscores with spaces; merge exact resulting duplicates",
        "task_index_mapping": task_lookup.tolist(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-converting-", dir=destination.parent))
    try:
        write_output(
            root,
            stage,
            paths,
            lengths,
            info,
            {STATE: state, ACTION: converted, TASK: task_indices},
            ranges,
            copy_videos,
            report,
        )
        write_tasks(tasks, stage)
        if protected.exists() or protected.is_symlink():
            raise FileExistsError(f"Destination/backup appeared during conversion: {protected}")
        if in_place:
            root.rename(backup)
            try:
                stage.rename(root)
            except BaseException:
                backup.rename(root)
                raise
            logging.info("Original dataset preserved at %s", backup)
        else:
            stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    logging.info(
        "Converted %d frames in %.2fs -> %s", info.total_frames, time.perf_counter() - started, destination
    )
    return destination


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="Exact local V3 dataset directory")
    parser.add_argument("--urdf", type=Path, required=True, help="Piper piper_description.urdf")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path, help="Default: <root>_ee_delta")
    output.add_argument(
        "--in-place", action="store_true", help="Replace root; preserve original as <root>_old"
    )
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument(
        "--joint-limit-margin-deg",
        type=float,
        default=5.0,
        help="Margin beyond URDF limits for detecting bad joints (default: 5 degrees)",
    )
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of trying hard links")
    parser.add_argument(
        "--gripper-encoding",
        choices=GRIPPER_ENCODINGS,
        default="signed",
        help="signed: -1=open/+1=closed; zero-one: 1=open/0=closed (default: signed)",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    convert_dataset(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
