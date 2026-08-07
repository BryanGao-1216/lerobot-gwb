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

"""Optional multi-RLDS input backend for ActionMem policies.

This module deliberately does not import TensorFlow, dlimp, or VQ-VLA at module
import time. Normal LeRobot dataset users therefore keep the existing dependency
and startup behavior. The external VQ-VLA checkout is used only for its OXE
schema registry and RLDS trajectory transforms; batches emitted here follow the
normal LeRobot policy contract.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, default_collate

from lerobot.configs.default import DatasetConfig
from lerobot.policies.actionmem.action_vqvae import (
    ActionVQVAEQ0Encoder,
    load_action_vqvae_q0_encoder,
)
from lerobot.utils.constants import ACTION, ACTION_TOKEN, OBS_STATE

_CAMERA_KEY_BY_VIEW = {
    "primary": "observation.images.image",
    "secondary": "observation.images.image2",
    "wrist": "observation.images.image3",
}
_ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


@dataclass(frozen=True)
class RLDSBackendModules:
    dl: Any
    tf: Any
    make_dataset_from_rlds: Callable[..., Any]
    apply_trajectory_transforms: Callable[..., Any]
    apply_per_dataset_frame_transforms: Callable[..., Any]
    apply_frame_transforms: Callable[..., Any]
    get_oxe_dataset_kwargs_and_weights: Callable[..., Any]
    named_mixtures: Mapping[str, list[tuple[str, float]]]
    normalization_type: Any


class RLDSDatasetMetadata:
    """Small metadata facade containing the attributes used by lerobot-train."""

    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path,
        features: dict[str, dict[str, Any]],
        stats: dict[str, dict[str, torch.Tensor]],
        total_frames: int,
        total_episodes: int,
        camera_keys: list[str],
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        self.features = features
        self.stats = stats
        self.total_frames = total_frames
        self.total_episodes = total_episodes
        self.camera_keys = camera_keys
        self.depth_keys: list[str] = []
        self.video_keys: list[str] = []
        self.image_keys = list(camera_keys)
        self.has_language_columns = False
        self.robot_type = "multi_rlds"
        self.fps = 1
        self.episodes = None
        # PreTrainedPolicy's model-card helper expects this LeRobot metadata
        # attribute even when the policy is not pushed to the Hub.
        self.tasks = SimpleNamespace(index=[])


class RLDSActionTokenCollator:
    """Default-collate an RLDS batch, then encode its raw action chunks to q0."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"rlds_q0_device={device!r} requested CUDA, but CUDA is not available in this process."
            )
        self.encoder: ActionVQVAEQ0Encoder = load_action_vqvae_q0_encoder(self.checkpoint_path)
        self.encoder.to(self.device)
        self.encoder.eval()

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = default_collate(samples)
        actions = batch[ACTION].to(device=self.device, dtype=torch.float32, non_blocking=True)
        with torch.inference_mode():
            q0_codes = self.encoder(actions)
        batch[ACTION_TOKEN] = q0_codes.to(device="cpu", dtype=torch.long)
        return batch


def _load_rlds_backend() -> RLDSBackendModules:
    try:
        import dlimp as dl
        import tensorflow as tf

        from lerobot.datasets.rlds import dataset as dataset_module, oxe as oxe_module
        from lerobot.datasets.rlds.utils import data_utils as utils_module
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The RLDS backend requires LeRobot's optional RLDS dependencies: TensorFlow, "
            "tensorflow-datasets, and dlimp. Install them with "
            "`pip install -e '.[rlds]'`, then install dlimp without its incompatible pinned "
            "dependencies using `pip install --no-deps "
            "'dlimp @ git+https://github.com/moojink/dlimp_openvla.git'`. "
            f"Missing import: {exc.name!r}."
        ) from exc

    return RLDSBackendModules(
        dl=dl,
        tf=tf,
        make_dataset_from_rlds=dataset_module.make_dataset_from_rlds,
        apply_trajectory_transforms=dataset_module.apply_trajectory_transforms,
        apply_per_dataset_frame_transforms=dataset_module.apply_per_dataset_frame_transforms,
        apply_frame_transforms=dataset_module.apply_frame_transforms,
        get_oxe_dataset_kwargs_and_weights=oxe_module.get_oxe_dataset_kwargs_and_weights,
        named_mixtures=oxe_module.OXE_NAMED_MIXTURES,
        normalization_type=utils_module.NormalizationType,
    )


def _load_mixture_spec(
    dataset_config: DatasetConfig,
    named_mixtures: Mapping[str, list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    if dataset_config.rlds_mixture_path is not None:
        path = Path(dataset_config.rlds_mixture_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RLDS mixture file does not exist: {path}")
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        entries = payload.get("datasets") if isinstance(payload, Mapping) else payload
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                "RLDS mixture JSON must be a non-empty list or contain a non-empty 'datasets' list."
            )
        mixture: list[tuple[str, float]] = []
        for entry in entries:
            if isinstance(entry, Mapping):
                name = entry.get("name")
                weight = entry.get("weight", 1.0)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                name, weight = entry
            else:
                raise ValueError(f"Invalid RLDS mixture entry: {entry!r}")
            if not isinstance(name, str) or not name:
                raise ValueError(f"RLDS dataset name must be a non-empty string, got {name!r}")
            weight = float(weight)
            if weight <= 0:
                raise ValueError(f"RLDS dataset weight must be positive, got {weight} for {name!r}")
            mixture.append((name, weight))
    else:
        mix_name = dataset_config.rlds_data_mix or dataset_config.repo_id
        mixture = list(named_mixtures.get(mix_name, [(mix_name, 1.0)]))

    seen: set[str] = set()
    for name, weight in mixture:
        if not isinstance(name, str) or not name:
            raise ValueError(f"RLDS dataset name must be a non-empty string, got {name!r}")
        numeric_weight = float(weight)
        if not np.isfinite(numeric_weight) or numeric_weight <= 0:
            raise ValueError(f"RLDS dataset weight must be finite and positive, got {weight} for {name!r}")
        if name in seen:
            raise ValueError(f"RLDS mixture contains duplicate dataset {name!r}.")
        seen.add(name)
    return [(name, float(weight)) for name, weight in mixture]


def _make_actionmem_standardizer(base_standardizer: Callable, dataset_name: str, tf: Any) -> Callable:
    """Reproduce the action extraction used to train the ActionMem VQ-VAE."""

    def standardize(trajectory: dict[str, Any]) -> dict[str, Any]:
        # DROID/RLBench VQ-VAE inputs were extracted as a target EEF pose
        # relative to the current EEF state. Preserve the raw tensors before
        # the OXE standardizer replaces/restructures them.
        raw_action = trajectory["action"]
        raw_observation = trajectory["observation"]
        if dataset_name == "rl_bench":
            raw_state = raw_observation.get("state", raw_observation.get("EEF_state"))
        else:
            raw_state = raw_observation.get("cartesian_position")

        trajectory = base_standardizer(trajectory)
        if dataset_name == "droid" or dataset_name.startswith("droid_"):
            if raw_state is None:
                raise ValueError(f"DROID dataset {dataset_name!r} has no observation.cartesian_position.")
            source_action = tf.cast(raw_action, tf.float32)
            source_state = tf.cast(raw_state, tf.float32)
            delta = source_action[..., :6] - source_state[..., :6]
            wrapped_rotation = tf.math.floormod(delta[..., 3:6] + np.pi, 2 * np.pi) - np.pi
            gripper = tf.where(
                source_action[..., 6:7] < 0.5,
                -tf.ones_like(source_action[..., 6:7]),
                tf.ones_like(source_action[..., 6:7]),
            )
            action = tf.concat([delta[..., :3], wrapped_rotation, gripper], axis=-1)
        elif dataset_name == "rl_bench":
            if raw_state is None:
                raise ValueError("RLBench has neither observation.state nor observation.EEF_state.")
            source_action = tf.cast(raw_action, tf.float32)
            source_state = tf.cast(raw_state, tf.float32)
            delta = source_action[..., :6] - source_state[..., :6]
            wrapped_rotation = tf.math.floormod(delta[..., 3:6] + np.pi, 2 * np.pi) - np.pi
            gripper = tf.where(
                tf.equal(source_action[..., 6:7], 0),
                tf.ones_like(source_action[..., 6:7]),
                -tf.ones_like(source_action[..., 6:7]),
            )
            action = tf.concat([delta[..., :3], wrapped_rotation, gripper], axis=-1)
        else:
            # OXE maps LIBERO open/close to 1/0; the VQ-VAE was trained on -1/+1.
            action = tf.cast(trajectory["action"], tf.float32)
            gripper = 1.0 - 2.0 * action[..., -1:]
            action = tf.concat([action[..., :-1], gripper], axis=-1)
        trajectory["action"] = action
        return trajectory

    return standardize


def _wrap_standardizers(per_dataset_kwargs: list[dict[str, Any]], action_transform: str, tf: Any) -> None:
    if action_transform == "identity":
        return
    for dataset_kwargs in per_dataset_kwargs:
        dataset_name = dataset_kwargs["name"]
        if not (
            dataset_name == "rl_bench"
            or dataset_name.startswith("droid")
            or dataset_name.startswith("libero")
        ):
            raise ValueError(
                "rlds_action_transform='actionmem' is only verified for DROID, RLBench, and LIBERO, "
                f"but the mixture contains {dataset_name!r}. Add its action conversion before training, "
                "or explicitly select --dataset.rlds_action_transform=identity after verifying that its "
                "7-D action convention matches the VQ-VAE."
            )
        base_standardizer = dataset_kwargs.get("standardize_fn")
        if base_standardizer is None:
            raise ValueError(f"RLDS dataset {dataset_name!r} has no OXE standardization transform.")
        dataset_kwargs["standardize_fn"] = _make_actionmem_standardizer(base_standardizer, dataset_name, tf)


def _denormalize_trajectory(trajectory: dict[str, Any], statistics: Mapping[str, Any], tf: Any):
    action_stats = statistics["action"]
    action_mask = tf.convert_to_tensor(
        action_stats.get("mask", np.ones_like(action_stats["mean"], dtype=bool)), dtype=tf.bool
    )
    action_mean = tf.convert_to_tensor(action_stats["mean"], dtype=tf.float32)
    action_std = tf.convert_to_tensor(action_stats["std"], dtype=tf.float32)
    trajectory["action"] = tf.where(
        action_mask,
        trajectory["action"] * (action_std + 1e-8) + action_mean,
        trajectory["action"],
    )

    observation = trajectory.get("observation", {})
    if "proprio" in observation and "proprio" in statistics:
        proprio_stats = statistics["proprio"]
        proprio_mean = tf.convert_to_tensor(proprio_stats["mean"], dtype=tf.float32)
        proprio_std = tf.convert_to_tensor(proprio_stats["std"], dtype=tf.float32)
        observation["proprio"] = observation["proprio"] * (proprio_std + 1e-8) + proprio_mean
    return trajectory


def _trim_invalid_chunk_targets(trajectory: dict[str, Any], horizon: int, tf: Any):
    valid_length = tf.maximum(tf.shape(trajectory["action"])[0] - horizon + 1, 0)
    return tf.nest.map_structure(lambda value: value[:valid_length], trajectory)


def _pad_trajectory_proprio(trajectory: dict[str, Any], state_dim: int, tf: Any):
    """Give every source the same state shape before TensorFlow interleaving."""
    proprio = trajectory["observation"]["proprio"]
    pad_width = state_dim - tf.shape(proprio)[-1]
    padded = tf.pad(proprio, [[0, 0], [0, pad_width]])
    # Preserve a static final dimension in element_spec. sample_from_datasets
    # requires compatible TensorSpecs across all sources.
    padded.set_shape(proprio.shape[:-1].concatenate([state_dim]))
    trajectory["observation"]["proprio"] = padded
    return trajectory


def _validate_statistics(
    dataset_name: str,
    statistics: Mapping[str, Any],
    *,
    action_dim: int,
    state_dim: int,
) -> None:
    if "action" not in statistics or "proprio" not in statistics:
        raise ValueError(f"RLDS dataset {dataset_name!r} must provide both action and proprio statistics.")
    actual_action_dim = np.asarray(statistics["action"]["mean"]).size
    if actual_action_dim != action_dim:
        raise ValueError(
            f"RLDS dataset {dataset_name!r} has action dimension {actual_action_dim}, but the "
            f"ActionMem VQ-VAE expects {action_dim}."
        )
    actual_state_dim = np.asarray(statistics["proprio"]["mean"]).size
    if actual_state_dim > state_dim:
        raise ValueError(
            f"RLDS dataset {dataset_name!r} has proprio dimension {actual_state_dim}, which exceeds "
            f"dataset.rlds_state_dim={state_dim}."
        )


def _pad_stat_vector(value: Any, size: int, fill: float = 0.0) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size > size:
        raise ValueError(f"RLDS feature dimension {array.size} exceeds configured state dimension {size}.")
    return np.pad(array, (0, size - array.size), constant_values=fill)


def _aggregate_weighted_stats(
    statistics: list[Mapping[str, Any]], weights: np.ndarray, feature: str, size: int
) -> dict[str, torch.Tensor]:
    means = np.stack([_pad_stat_vector(stats[feature]["mean"], size) for stats in statistics])
    stds = np.stack([_pad_stat_vector(stats[feature]["std"], size) for stats in statistics])
    mean = np.sum(weights[:, None] * means, axis=0)
    second_moment = np.sum(weights[:, None] * (stds**2 + means**2), axis=0)
    variance = np.maximum(second_moment - mean**2, 0.0)

    minimum = np.min(
        np.stack([_pad_stat_vector(stats[feature]["min"], size) for stats in statistics]), axis=0
    )
    maximum = np.max(
        np.stack([_pad_stat_vector(stats[feature]["max"], size) for stats in statistics]), axis=0
    )
    q01 = np.sum(
        weights[:, None]
        * np.stack(
            [_pad_stat_vector(stats[feature].get("q01", stats[feature]["min"]), size) for stats in statistics]
        ),
        axis=0,
    )
    q99 = np.sum(
        weights[:, None]
        * np.stack(
            [_pad_stat_vector(stats[feature].get("q99", stats[feature]["max"]), size) for stats in statistics]
        ),
        axis=0,
    )
    count = sum(int(stats.get("num_transitions", 0)) for stats in statistics)
    return {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "std": torch.from_numpy(np.sqrt(variance).astype(np.float32)),
        "min": torch.from_numpy(minimum.astype(np.float32)),
        "max": torch.from_numpy(maximum.astype(np.float32)),
        "q01": torch.from_numpy(q01.astype(np.float32)),
        "q99": torch.from_numpy(q99.astype(np.float32)),
        "count": torch.tensor([count], dtype=torch.long),
    }


def _image_stats() -> dict[str, torch.Tensor]:
    # Matches lerobot.utils.constants.IMAGENET_STATS without making the RLDS
    # adapter depend on the setting applied inside the LeRobot-only factory.
    return {
        "mean": torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
        "std": torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
        "min": torch.zeros(3, dtype=torch.float32),
        "max": torch.ones(3, dtype=torch.float32),
    }


def _resolve_rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


class ActionMemRLDSDataset(IterableDataset):
    """Weighted multi-RLDS stream adapted to the ActionMem batch contract."""

    def __init__(
        self,
        *,
        dataset_config: DatasetConfig,
        action_horizon: int,
        action_dim: int,
        state_dim: int,
        action_vqvae_checkpoint_path: str | Path,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset_config = dataset_config
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.backend = _load_rlds_backend()
        self.rank, self.world_size = _resolve_rank_and_world_size()
        self.backend.tf.random.set_seed(seed)

        mixture_spec = _load_mixture_spec(dataset_config, self.backend.named_mixtures)
        per_dataset_kwargs, base_weights = self.backend.get_oxe_dataset_kwargs_and_weights(
            Path(dataset_config.root),
            mixture_spec,
            load_camera_views=tuple(dataset_config.rlds_camera_views),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=self.backend.normalization_type.NORMAL,
        )
        if len(per_dataset_kwargs) != len(mixture_spec):
            loaded = {kwargs["name"] for kwargs in per_dataset_kwargs}
            missing = [name for name, _ in mixture_spec if name not in loaded]
            raise ValueError(f"The VQ-VLA OXE registry could not load RLDS datasets: {missing}")
        if len(base_weights) != len(per_dataset_kwargs):
            raise ValueError(
                "The VQ-VLA OXE registry returned a different number of weights and datasets: "
                f"{len(base_weights)} weights for {len(per_dataset_kwargs)} datasets."
            )
        _wrap_standardizers(per_dataset_kwargs, dataset_config.rlds_action_transform, self.backend.tf)

        self._dataset, all_statistics, effective_weights, valid_sizes = self._build_interleaved_dataset(
            per_dataset_kwargs,
            np.asarray(base_weights, dtype=np.float64),
        )
        self.dataset_names = [kwargs["name"] for kwargs in per_dataset_kwargs]
        self.sample_weights = effective_weights
        self.num_frames = int(
            max(size / weight for size, weight in zip(valid_sizes, effective_weights, strict=True))
        )
        self.num_episodes = sum(int(stats.get("num_trajectories", 0)) for stats in all_statistics)
        self.episodes = None
        self.absolute_to_relative_idx = None

        camera_keys = [_CAMERA_KEY_BY_VIEW[view] for view in dataset_config.rlds_camera_views]
        features: dict[str, dict[str, Any]] = {
            OBS_STATE: {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"state_{index}" for index in range(state_dim)],
            },
            ACTION: {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": _ACTION_NAMES[:action_dim],
            },
        }
        for key in camera_keys:
            features[key] = {
                "dtype": "image",
                "shape": (*dataset_config.rlds_resize_size, 3),
                "names": ["height", "width", "channels"],
            }

        stats = {
            ACTION: _aggregate_weighted_stats(all_statistics, effective_weights, "action", action_dim),
            OBS_STATE: _aggregate_weighted_stats(all_statistics, effective_weights, "proprio", state_dim),
        }
        for key in camera_keys:
            stats[key] = _image_stats()
        mix_name = dataset_config.rlds_data_mix or dataset_config.repo_id
        self.meta = RLDSDatasetMetadata(
            repo_id=mix_name,
            root=dataset_config.root or "",
            features=features,
            stats=stats,
            total_frames=self.num_frames,
            total_episodes=self.num_episodes,
            camera_keys=camera_keys,
        )
        self.collate_fn = RLDSActionTokenCollator(
            checkpoint_path=action_vqvae_checkpoint_path,
            device=dataset_config.rlds_q0_device,
        )

        logging.info(
            "RLDS mixture on rank %d/%d: %s",
            self.rank,
            self.world_size,
            ", ".join(
                f"{name}={weight:.4f}"
                for name, weight in zip(self.dataset_names, effective_weights, strict=True)
            ),
        )

    def _build_interleaved_dataset(
        self,
        per_dataset_kwargs: list[dict[str, Any]],
        base_weights: np.ndarray,
    ) -> tuple[Any, list[Mapping[str, Any]], np.ndarray, list[int]]:
        all_statistics: list[Mapping[str, Any]] = []
        valid_sizes: list[int] = []
        for kwargs in per_dataset_kwargs:
            statistics_kwargs = copy.deepcopy(kwargs)
            statistics_kwargs.pop("dataset_frame_transform_kwargs", None)
            _, statistics = self.backend.make_dataset_from_rlds(
                **statistics_kwargs,
                train=True,
                num_parallel_calls=self.dataset_config.rlds_num_parallel_calls,
                num_parallel_reads=self.dataset_config.rlds_num_parallel_calls,
            )
            _validate_statistics(
                kwargs["name"], statistics, action_dim=self.action_dim, state_dim=self.state_dim
            )
            all_statistics.append(statistics)
            valid_sizes.append(
                max(
                    int(statistics["num_transitions"])
                    - (self.action_horizon - 1) * int(statistics["num_trajectories"]),
                    1,
                )
            )

        effective_weights = base_weights.copy()
        if self.dataset_config.rlds_balance_weights:
            effective_weights *= np.asarray(valid_sizes, dtype=np.float64)
        effective_weights /= effective_weights.sum()

        datasets = []
        threads_per_dataset = max(
            1, self.dataset_config.rlds_num_parallel_calls // max(len(per_dataset_kwargs), 1)
        )
        for kwargs, statistics in zip(per_dataset_kwargs, all_statistics, strict=True):
            source_kwargs = copy.deepcopy(kwargs)
            frame_transform_kwargs = source_kwargs.pop("dataset_frame_transform_kwargs", {})
            # DROID's original zero-action filter assumes normalized actions.
            # This backend explicitly restores raw actions before chunking.
            frame_transform_kwargs.pop("chunk_filter_fn", None)
            source, _ = self.backend.make_dataset_from_rlds(
                **source_kwargs,
                train=True,
                num_parallel_calls=threads_per_dataset,
                num_parallel_reads=threads_per_dataset,
                dataset_statistics=statistics,
            )
            if self.world_size > 1:
                source = source.shard(self.world_size, self.rank)
            source = source.traj_map(
                partial(_denormalize_trajectory, statistics=statistics, tf=self.backend.tf),
                threads_per_dataset,
            )
            source = source.traj_map(
                partial(_pad_trajectory_proprio, state_dim=self.state_dim, tf=self.backend.tf),
                threads_per_dataset,
            )
            source = self.backend.apply_trajectory_transforms(
                source.repeat(),
                train=True,
                window_size=1,
                future_action_window_size=self.action_horizon - 1,
                skip_unlabeled=self.dataset_config.rlds_skip_unlabeled,
                goal_relabeling_strategy=None,
                num_parallel_calls=threads_per_dataset,
            )
            source = source.traj_map(
                partial(_trim_invalid_chunk_targets, horizon=self.action_horizon, tf=self.backend.tf),
                threads_per_dataset,
            ).flatten(num_parallel_calls=threads_per_dataset)
            source = self.backend.apply_per_dataset_frame_transforms(source, **frame_transform_kwargs)
            datasets.append(source)

        mixed = self.backend.dl.DLataset.sample_from_datasets(datasets, effective_weights)
        mixed = mixed.shuffle(self.dataset_config.rlds_shuffle_buffer_size)
        mixed = self.backend.apply_frame_transforms(
            mixed,
            train=True,
            resize_size=tuple(self.dataset_config.rlds_resize_size),
            num_parallel_calls=self.dataset_config.rlds_num_parallel_calls,
        )
        return mixed.with_ram_budget(1), all_statistics, effective_weights, valid_sizes

    def _to_lerobot_sample(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        observation = frame["observation"]
        state = np.asarray(observation["proprio"][-1], dtype=np.float32).reshape(-1)
        if state.size > self.state_dim:
            raise ValueError(
                f"RLDS proprio dimension {state.size} exceeds configured state dimension {self.state_dim}."
            )
        state = np.pad(state, (0, self.state_dim - state.size))

        action = np.asarray(frame["action"], dtype=np.float32)
        if action.shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"Expected RLDS action chunk {(self.action_horizon, self.action_dim)}, got {action.shape}."
            )
        sample: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(state),
            ACTION: torch.from_numpy(action.copy()),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "task": _decode_text(frame["task"]["language_instruction"]),
            "dataset_name": _decode_text(frame["dataset_name"]),
        }

        pad_masks = observation.get("pad_mask_dict", {})
        for view in self.dataset_config.rlds_camera_views:
            source_key = f"image_{view}"
            target_key = _CAMERA_KEY_BY_VIEW[view]
            image = np.asarray(observation[source_key][-1], dtype=np.uint8)
            sample[target_key] = torch.from_numpy(np.moveaxis(image, -1, 0).copy())
            mask_value = pad_masks.get(source_key, np.asarray([True]))
            mask = bool(np.asarray(mask_value).reshape(-1)[-1])
            sample[f"{target_key}_padding_mask"] = torch.tensor(mask, dtype=torch.bool)
        return sample

    def __iter__(self):
        for frame in self._dataset.as_numpy_iterator():
            yield self._to_lerobot_sample(frame)

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int):
        raise NotImplementedError("ActionMemRLDSDataset is an infinite IterableDataset.")


def resolve_actionmem_token_metadata(policy_config: Any) -> SimpleNamespace:
    """Resolve the token map without importing the heavy policy model."""
    token_map_path = getattr(policy_config, "action_token_map_path", None)
    pretrained_path = getattr(policy_config, "pretrained_path", None)
    companion_path = (
        Path(pretrained_path).expanduser() / "token_map.json" if pretrained_path is not None else None
    )
    if companion_path is not None and companion_path.is_file():
        token_map_path = str(companion_path.resolve())
        policy_config.action_token_map_path = token_map_path

    if policy_config.type == "smol_actionmem":
        from lerobot.policies.smol_actionmem.tokenization_smol_actionmem import SmolActionMemTokenMap

        return SmolActionMemTokenMap.from_json(token_map_path)
    if policy_config.type == "smol_actionmem2":
        from lerobot.policies.smol_actionmem2.tokenization_smol_actionmem import SmolActionMem2TokenMap

        return SmolActionMem2TokenMap.from_json(token_map_path)
    if policy_config.type == "actionmem":
        from lerobot.policies.actionmem.tokenization_actionmem import ActionMemTokenMap

        return ActionMemTokenMap.from_json(token_map_path)
    if policy_config.type == "pi05_actionmem":
        from lerobot.policies.pi05_actionmem.tokenization_pi05_actionmem import PI05ActionMemTokenMap

        return PI05ActionMemTokenMap.from_json(token_map_path)
    raise ValueError(f"RLDS ActionMem backend does not support policy type {policy_config.type!r}.")


def make_actionmem_rlds_dataset(cfg: Any) -> ActionMemRLDSDataset:
    token_metadata = resolve_actionmem_token_metadata(cfg.trainable_config)
    checkpoint_path = (
        cfg.dataset.rlds_action_vqvae_checkpoint_path
        or getattr(cfg.trainable_config, "action_vqvae_checkpoint_path", None)
        or token_metadata.vqvae_checkpoint_path
    )
    if checkpoint_path is None:
        raise ValueError(
            "RLDS action-token generation needs a VQ-VAE checkpoint. Set "
            "--dataset.rlds_action_vqvae_checkpoint_path or policy.action_vqvae_checkpoint_path, "
            "or store checkpoint_path in token_map.json."
        )
    action_horizon = int(token_metadata.action_horizon)
    action_dim = int(token_metadata.action_dim)
    configured_chunk_size = int(getattr(cfg.trainable_config, "chunk_size", action_horizon))
    if configured_chunk_size != action_horizon:
        raise ValueError(
            f"Policy chunk_size={configured_chunk_size} must match VQ-VAE horizon={action_horizon}."
        )
    state_dim = cfg.dataset.rlds_state_dim or int(getattr(cfg.trainable_config, "max_state_dim", 32))
    return ActionMemRLDSDataset(
        dataset_config=cfg.dataset,
        action_horizon=action_horizon,
        action_dim=action_dim,
        state_dim=state_dim,
        action_vqvae_checkpoint_path=checkpoint_path,
        seed=cfg.seed if cfg.seed is not None else 0,
    )
