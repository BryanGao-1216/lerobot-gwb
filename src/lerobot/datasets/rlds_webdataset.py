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

"""Local OpenX WebDataset tar reader used by the RLDS/OXE training backend.

The ``jxu124/OpenX-Embodiment`` release stores one pickled RLDS episode per
``*.data.pickle`` tar member. This module streams those members in place and
turns the episode's ``steps`` list back into a tensor trajectory. It does not
extract archives and deliberately leaves dataset-specific semantics to the
vendored VQ-VLA OXE standardizers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import tarfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Directory aliases used by the local OpenX tar release and older mixtures.
_OPENX_DIRECTORY_ALIASES = {
    "bridge_orig": "bridge",
    "fmb": "fmb_dataset",
}

logger = logging.getLogger(__name__)


def resolve_openx_tar_paths(root: str | Path, dataset_name: str) -> tuple[Path, ...]:
    """Return the sorted local tar shards for one OXE source."""
    root = Path(root).expanduser()
    candidates = [root / dataset_name]
    alias = _OPENX_DIRECTORY_ALIASES.get(dataset_name)
    if alias is not None:
        candidates.append(root / alias)

    for dataset_dir in candidates:
        paths = tuple(sorted(dataset_dir.glob("*.tar")))
        if paths:
            return paths
    return ()


def openx_tar_manifest(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    """Build a cache key that changes when local shards are replaced."""
    return tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)


def iter_openx_tar_episodes(
    paths: Sequence[Path],
    *,
    seed: int = 0,
    shuffle_shards: bool = False,
) -> Iterator[Mapping[str, Any]]:
    """Yield trusted local OpenX episode payloads without extracting tar files."""
    ordered_paths = list(paths)
    if shuffle_shards:
        random.Random(seed).shuffle(ordered_paths)

    for path in ordered_paths:
        try:
            with tarfile.open(path, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".data.pickle"):
                        continue
                    fileobj = archive.extractfile(member)
                    if fileobj is None:
                        raise RuntimeError(f"Could not read {member.name!r} from OpenX shard {path}.")
                    try:
                        # This dataset format is Python pickle by design. Only local,
                        # trusted OpenX shards should be supplied as dataset.root.
                        payload = pickle.loads(fileobj.read())  # noqa: S301  # nosec B301
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to deserialize {member.name!r} from OpenX shard {path}: {exc}"
                        ) from exc
                    if not isinstance(payload, Mapping) or not isinstance(payload.get("steps"), list):
                        raise ValueError(
                            f"OpenX member {member.name!r} in {path} must contain a mapping with a 'steps' list."
                        )
                    if payload["steps"]:
                        yield payload
        except (tarfile.TarError, OSError) as exc:
            raise RuntimeError(f"Failed to open OpenX WebDataset shard {path}: {exc}") from exc


def _is_encoded_image(value: Any) -> bool:
    return isinstance(value, Mapping) and "bytes" in value and set(value).issubset({"bytes", "path"})


def _encoded_image_bytes(value: Mapping[str, Any], path: str) -> bytes:
    data = value.get("bytes")
    if data is not None:
        return bytes(data)
    image_path = value.get("path")
    if image_path:
        return Path(image_path).read_bytes()
    raise ValueError(f"Encoded image at {path} has neither bytes nor a readable path.")


def _stack_step_values(values: Sequence[Any], path: str) -> Any:
    first = values[0]
    if all(_is_encoded_image(value) for value in values):
        return np.asarray(
            [_encoded_image_bytes(value, path) for value in values],
            dtype=object,
        )

    if isinstance(first, Mapping):
        expected_keys = set(first)
        for index, value in enumerate(values[1:], start=1):
            if not isinstance(value, Mapping) or set(value) != expected_keys:
                raise ValueError(f"OpenX episode field {path} changes mapping keys at step {index}.")
        stacked = {}
        for key in first:
            nested_values = [value[key] for value in values]
            if all(value is None for value in nested_values):
                continue
            if any(value is None for value in nested_values):
                raise ValueError(f"OpenX episode field {path}/{key} is None only at some steps.")
            stacked[key] = _stack_step_values(nested_values, f"{path}/{key}")
        return stacked

    if isinstance(first, str):
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"OpenX episode field {path} mixes string and non-string values.")
        return np.asarray(values, dtype=object)
    if isinstance(first, (bytes, bytearray, memoryview)):
        if not all(isinstance(value, (bytes, bytearray, memoryview)) for value in values):
            raise ValueError(f"OpenX episode field {path} mixes byte and non-byte values.")
        return np.asarray([bytes(value) for value in values], dtype=object)

    # Some versions of the release store decoded images as PIL objects.
    if first.__class__.__module__.startswith("PIL."):
        values = [np.asarray(value) for value in values]

    try:
        return np.stack([np.asarray(value) for value in values])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OpenX episode field {path} cannot be stacked into a dense tensor.") from exc


def stack_openx_episode_steps(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one WebDataset episode's step list into an RLDS-style trajectory."""
    steps = payload["steps"]
    if not steps or not all(isinstance(step, Mapping) for step in steps):
        raise ValueError("OpenX episode 'steps' must be a non-empty list of mappings.")
    expected_keys = set(steps[0])
    if any(set(step) != expected_keys for step in steps[1:]):
        raise ValueError("OpenX episode top-level step keys are inconsistent.")
    return {
        key: _stack_step_values([step[key] for step in steps], key)
        for key in steps[0]
        if not all(step[key] is None for step in steps)
    }


def _trajectory_to_tensors(trajectory: Mapping[str, Any], tf: Any) -> dict[str, Any]:
    """Convert a stacked NumPy trajectory to eager tensors accepted by OXE transforms."""

    def convert(value: Any):
        if isinstance(value, Mapping):
            return {key: convert(nested) for key, nested in value.items()}
        array = np.asarray(value)
        if array.dtype.kind in {"O", "S", "U"}:
            # Going through a Python list avoids NumPy object-array conversion
            # differences between TensorFlow versions.
            return tf.convert_to_tensor(array.tolist(), dtype=tf.string)
        return tf.convert_to_tensor(array)

    return {key: convert(value) for key, value in trajectory.items()}


def transform_openx_tar_episode(
    payload: Mapping[str, Any],
    *,
    tf: Any,
    transform: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Run an OXE eager transform on one pickled tar episode."""
    return transform(_trajectory_to_tensors(stack_openx_episode_steps(payload), tf))


def _statistics_cache_paths(paths: Sequence[Path], hash_dependencies: Sequence[str]) -> tuple[Path, Path]:
    unique_hash = hashlib.sha256(
        "".join(hash_dependencies).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    filename = f"dataset_statistics_{unique_hash}.json"
    return paths[0].parent / filename, Path.home() / ".cache" / "lerobot" / "rlds" / filename


def _load_cached_statistics(paths: Sequence[Path]) -> dict[str, Any] | None:
    for path in paths:
        if path.is_file():
            logger.info("Loading existing OpenX tar statistics from %s.", path)
            with path.open(encoding="utf-8") as stream:
                return json.load(stream)
    return None


def _write_statistics_cache(metadata: Mapping[str, Any], primary: Path, fallback: Path) -> Path:
    for path in (primary, fallback):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            temporary.replace(path)
            return path
        except OSError as exc:
            logger.warning("Could not write OpenX tar statistics to %s: %s", path, exc)
    raise OSError(f"Could not cache OpenX tar statistics at {primary} or {fallback}.")


def load_or_compute_openx_tar_statistics(
    *,
    paths: Sequence[Path],
    tf: Any,
    restructure_fn: Callable[[dict[str, Any]], Mapping[str, Any]],
    hash_dependencies: Sequence[str],
) -> dict[str, Any]:
    """Compute VQ-VLA-compatible action/proprio statistics directly from tar episodes.

    This deliberately avoids feeding the Python tar generator back through a
    TensorFlow iterator. Only the small action and proprio arrays are retained;
    compressed image bytes are discarded after each OXE standardizer call.
    """
    if not paths:
        raise ValueError("At least one OpenX tar shard is required to compute statistics.")
    primary_cache, fallback_cache = _statistics_cache_paths(paths, hash_dependencies)
    cached = _load_cached_statistics((primary_cache, fallback_cache))
    if cached is not None:
        return cached

    logger.info(
        "Computing OpenX tar statistics from %d shard(s). This one-time scan does not extract files.",
        len(paths),
    )
    actions: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    num_transitions = 0
    num_trajectories = 0
    for payload in iter_openx_tar_episodes(paths):
        raw_trajectory = stack_openx_episode_steps(payload)
        trajectory = restructure_fn(_trajectory_to_tensors(raw_trajectory, tf))
        action = np.asarray(trajectory["action"], dtype=np.float32)
        observation = trajectory.get("observation", {})
        proprio_value = observation.get("proprio") if isinstance(observation, Mapping) else None
        proprio = (
            np.zeros_like(action) if proprio_value is None else np.asarray(proprio_value, dtype=np.float32)
        )
        if action.ndim != 2 or proprio.ndim != 2 or action.shape[0] != proprio.shape[0]:
            raise ValueError(
                "OXE standardizer must produce rank-2 action/proprio tensors with the same time dimension; "
                f"got action={action.shape}, proprio={proprio.shape}."
            )
        actions.append(action)
        proprios.append(proprio)
        num_transitions += action.shape[0]
        num_trajectories += 1
        if num_trajectories % 100 == 0:
            logger.info(
                "Scanned %d OpenX tar episodes (%d transitions) for statistics.",
                num_trajectories,
                num_transitions,
            )

    if not actions:
        raise ValueError("OpenX tar shards contain no non-empty episodes for statistics.")
    all_actions = np.concatenate(actions, axis=0)
    all_proprios = np.concatenate(proprios, axis=0)

    def summarize(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "max": values.max(0).tolist(),
            "min": values.min(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    metadata = {
        "action": summarize(all_actions),
        "proprio": summarize(all_proprios),
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }
    cache_path = _write_statistics_cache(metadata, primary_cache, fallback_cache)
    logger.info(
        "Computed OpenX tar statistics for %d episodes (%d transitions); cached at %s.",
        num_trajectories,
        num_transitions,
        cache_path,
    )
    return metadata
