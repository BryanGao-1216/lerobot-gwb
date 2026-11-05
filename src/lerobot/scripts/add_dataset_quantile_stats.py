#!/usr/bin/env python
r"""Add exact q01/q10/q50/q90/q99 statistics to a local LeRobot V3 dataset.

Run with the Python environment where this local LeRobot checkout is installed::

    python -m lerobot.scripts.add_dataset_quantile_stats \
        --root /path/to/Datasets/libero-90/libero_90

By default, process every numeric scalar/vector feature, including action, state,
other numeric observations, and numeric index/timestamp fields. Select a subset::

    python -m lerobot.scripts.add_dataset_quantile_stats \
        --root /path/to/libero_90 --features action observation.state

Only missing quantiles are added. --overwrite replaces existing quantiles, too.
Existing min/max/mean/std/count values are preserved. Quantiles are calculated
from all samples using NumPy's linear interpolation, never by averaging episode
quantiles. Both meta/stats.json and meta/episodes/**/*.parquet are updated.

Only numeric data columns are read; no images/videos are decoded, rewritten, or
uploaded. Numeric columns are loaded once, with one feature at a time converted
to float64 for calculation. RAM usage scales with the numeric part of the dataset.

Metadata is prepared in a temporary directory and then installed by directory
rename. The original meta/ directory is retained as meta_before_quantiles_<id>/
under the dataset root. --dry-run computes results without writing anything.
"""

import argparse
import logging
import math
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError("Install dataset dependencies with: uv sync --locked --extra dataset") from exc

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES
from lerobot.datasets.io_utils import write_stats
from lerobot.utils.io_utils import load_json

QUANTILE_KEYS = [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]


def is_numeric(feature: dict) -> bool:
    try:
        return np.dtype(feature["dtype"]).kind in "biuf"
    except TypeError:
        return False


def numeric_array(table: pa.Table, key: str, shape: tuple) -> np.ndarray:
    """Read scalar/list/fixed-size-list columns without creating Python rows."""
    column = table[key].combine_chunks()
    if column.null_count:
        raise ValueError(f"Null rows in {key}.")
    if (
        pa.types.is_list(column.type)
        or pa.types.is_large_list(column.type)
        or pa.types.is_fixed_size_list(column.type)
    ):
        if not np.all(column.value_lengths().to_numpy() == math.prod(shape)):
            raise ValueError(f"Values in {key} do not match feature shape {shape}.")
        column = column.flatten()
    elif math.prod(shape) != 1:
        raise ValueError(f"Scalar column {key} does not match feature shape {shape}.")
    if column.null_count:
        raise ValueError(f"Null components in {key}.")
    values = column.to_numpy(zero_copy_only=False).reshape(len(table), -1).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite samples in {key}; cannot compute exact quantiles.")
    return values


def add_quantile_stats(root: str | Path, features=None, overwrite=False, dry_run=False) -> Path | None:
    started = time.perf_counter()
    root = Path(root).expanduser().resolve()
    info = load_json(root / "meta/info.json")
    stats = load_json(root / "meta/stats.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("This script requires a local LeRobot v3.0 dataset.")

    selected = (
        list(dict.fromkeys(features))
        if features
        else [key for key, feature in info["features"].items() if is_numeric(feature)]
    )
    if not selected:
        raise ValueError("No numeric features selected.")
    for key in selected:
        feature = info["features"].get(key)
        if feature is None or not is_numeric(feature):
            raise ValueError(f"Unknown/non-numeric feature: {key}")
        if len(feature["shape"]) > 1:
            raise ValueError(f"Only numeric scalars/vectors are supported: {key}")
        if key not in stats:
            raise ValueError(f"Missing basic statistics for {key}; compute those first.")

    episode_paths = sorted((root / "meta/episodes").glob("*/*.parquet"))
    if not episode_paths:
        raise ValueError("No V3 episode metadata found.")
    episode_tables = [pq.read_table(path) for path in episode_paths]
    selected = [
        key
        for key in selected
        if overwrite
        or any(q not in stats[key] for q in QUANTILE_KEYS)
        or any(
            f"stats/{key}/{q}" not in table.column_names for table in episode_tables for q in QUANTILE_KEYS
        )
    ]
    if not selected:
        logging.info(
            "All selected global and per-episode quantiles already exist. Use --overwrite to recompute."
        )
        return None

    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise ValueError("No V3 data parquet files found.")
    columns = list(dict.fromkeys(["episode_index", "index", *selected]))
    table = pa.concat_tables(
        [pq.read_table(path, columns=columns) for path in tqdm(paths, desc="Reading numeric data")]
    )
    if not len(table) or len(table) != info["total_frames"]:
        raise ValueError("Data frame count differs from info.json.")
    if not np.array_equal(table["index"].to_numpy(), np.arange(len(table))):
        raise ValueError("Data files must contain all frames in contiguous global-index order.")
    episode_ids = table["episode_index"].to_numpy()
    ids, begins, counts = np.unique(episode_ids, return_index=True, return_counts=True)
    if len(ids) != info["total_episodes"] or np.any(np.diff(episode_ids) < 0):
        raise ValueError("Episode count/order differs from info.json.")
    row_for_episode = {int(ep): row for row, ep in enumerate(ids)}
    seen = set()
    for metadata in episode_tables:
        for row in metadata.select(
            ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
        ).to_pylist():
            ep = row["episode_index"]
            if ep in seen or ep not in row_for_episode:
                raise ValueError(f"Unexpected/duplicate episode metadata: {ep}")
            seen.add(ep)
            i = row_for_episode[ep]
            if (row["length"], row["dataset_from_index"], row["dataset_to_index"]) != (
                counts[i],
                begins[i],
                begins[i] + counts[i],
            ):
                raise ValueError(f"Episode {ep} has inconsistent frame bounds.")
    if seen != row_for_episode.keys():
        raise ValueError("Episode metadata does not cover all data episodes.")

    per_episode = {}
    for key in tqdm(selected, desc="Exact quantiles"):
        shape = tuple(info["features"][key]["shape"])
        values = numeric_array(table, key, shape)
        global_quantiles = np.quantile(values, DEFAULT_QUANTILES, axis=0, method="linear")
        for q_key, quantile in zip(QUANTILE_KEYS, global_quantiles, strict=True):
            if overwrite or q_key not in stats[key]:
                stats[key][q_key] = quantile.reshape(shape).tolist()
        per_episode[key] = np.stack(
            [
                np.quantile(values[begin : begin + count], DEFAULT_QUANTILES, axis=0, method="linear")
                for begin, count in zip(begins, counts, strict=True)
            ]
        )
        del values
    logging.info(
        "Computed %d features, %d frames, %d episodes in %.2fs",
        len(selected),
        len(table),
        len(ids),
        time.perf_counter() - started,
    )
    if dry_run:
        logging.info("Dry run: metadata was not modified.")
        return None

    stage = Path(tempfile.mkdtemp(prefix=".quantile-stats-", dir=root))
    backup = root / f"meta_before_quantiles_{uuid.uuid4().hex[:12]}"
    try:
        shutil.copytree(root / "meta", stage / "meta")
        write_stats(stats, stage)
        for path, metadata in tqdm(
            list(zip(episode_paths, episode_tables, strict=True)), desc="Writing metadata"
        ):
            rows = [row_for_episode[ep] for ep in metadata["episode_index"].to_pylist()]
            for key in selected:
                for q_index, q_key in enumerate(QUANTILE_KEYS):
                    name = f"stats/{key}/{q_key}"
                    index = metadata.schema.get_field_index(name)
                    if index >= 0 and not overwrite:
                        continue
                    column = pa.array(per_episode[key][rows, q_index].tolist())
                    if index >= 0:
                        metadata = metadata.set_column(index, name, column)
                    else:
                        metadata = metadata.append_column(name, column)
            # Pandas/HF annotations describe the old columns; retain other metadata.
            annotations = {
                k: v
                for k, v in (metadata.schema.metadata or {}).items()
                if k not in {b"pandas", b"huggingface"}
            }
            pq.write_table(
                metadata.replace_schema_metadata(annotations or None),
                stage / path.relative_to(root),
                compression="snappy",
            )
        (root / "meta").rename(backup)
        try:
            (stage / "meta").rename(root / "meta")
        except BaseException:
            backup.rename(root / "meta")
            raise
    finally:
        shutil.rmtree(stage)
    logging.info("Updated %s; original metadata preserved at %s", root / "meta", backup)
    return backup


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="Exact local V3 dataset directory")
    parser.add_argument("--features", nargs="+", help="Default: all numeric scalar/vector features")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing quantiles, preserving basic stats"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and validate without writing metadata"
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    add_quantile_stats(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
