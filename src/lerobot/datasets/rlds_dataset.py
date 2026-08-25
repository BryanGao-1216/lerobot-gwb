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

This module deliberately does not import TensorFlow or dlimp at module import
time. Normal LeRobot dataset users therefore keep the existing dependency and
startup behavior. VQ-VLA's vendored OXE schema registry and RLDS trajectory
transforms define the action contract; batches emitted here follow the normal
LeRobot policy contract.
"""

from __future__ import annotations

import copy
import io
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
from PIL import Image
from torch.utils.data import IterableDataset, default_collate

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.rlds_webdataset import (
    iter_openx_tar_episodes,
    resolve_openx_tar_paths,
    transform_openx_tar_episode,
)
from lerobot.policies.effect_tokenizer import (
    load_effect_tokenizer_metadata,
    load_effect_vqvae_action_encoder,
)
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN,
    ACTION_TOKEN_DISTANCES,
    ACTION_TOKENIZER_INPUT,
    OBS_STATE,
)

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
    make_dataset_from_webdataset: Callable[..., Any]
    make_restructure_fn: Callable[..., Any]
    apply_trajectory_transforms: Callable[..., Any]
    apply_per_dataset_frame_transforms: Callable[..., Any]
    apply_frame_transforms: Callable[..., Any]
    get_oxe_dataset_kwargs_and_weights: Callable[..., Any]
    named_mixtures: Mapping[str, list[tuple[str, float]]]
    standardization_transforms: Mapping[str, Callable[..., Any]]
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
        fps: float,
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
        self.fps = fps
        self.episodes = None
        # PreTrainedPolicy's model-card helper expects this LeRobot metadata
        # attribute even when the policy is not pushed to the Hub.
        self.tasks = SimpleNamespace(index=[])


class RLDSActionTokenCollator:
    """Collate RLDS samples and encode normalized chunks into action codes."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        expected_horizon: int | None = None,
        expected_action_dim: int | None = None,
        expected_codebook_size: int | None = None,
        expected_target_control_hz: float | None = None,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"rlds_action_tokenizer_device={device!r} requested CUDA, but CUDA is not available."
            )
        self.encoder = load_effect_vqvae_action_encoder(self.checkpoint_path)

        expected_values = {
            "horizon": expected_horizon,
            "action_dim": expected_action_dim,
            "codebook_size": expected_codebook_size,
        }
        mismatches = [
            f"{name}: checkpoint={getattr(self.encoder, name)}, expected={expected}"
            for name, expected in expected_values.items()
            if expected is not None and int(getattr(self.encoder, name)) != int(expected)
        ]
        checkpoint_hz = getattr(self.encoder, "target_control_hz", None)
        target_hz_mismatch = (checkpoint_hz is None) != (expected_target_control_hz is None)
        if checkpoint_hz is not None and expected_target_control_hz is not None:
            target_hz_mismatch = not np.isclose(checkpoint_hz, expected_target_control_hz)
        if target_hz_mismatch:
            mismatches.append(
                f"target_control_hz: checkpoint={checkpoint_hz}, expected={expected_target_control_hz}"
            )
        if mismatches:
            raise ValueError(
                "Action-tokenizer checkpoint is incompatible with the RLDS/policy contract: "
                + "; ".join(mismatches)
            )
        self.encoder.to(self.device)
        self.encoder.eval()

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = default_collate(samples)
        actions = batch[ACTION_TOKENIZER_INPUT].to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        with torch.inference_mode():
            code_distances = self.encoder.compute_code_distances(actions)
            codes = code_distances.argmin(dim=-1)
        batch[ACTION_TOKEN] = codes.to(device="cpu", dtype=torch.long)
        batch[ACTION_TOKEN_DISTANCES] = code_distances.to(device="cpu", dtype=torch.float32)
        return batch


def _load_rlds_backend() -> RLDSBackendModules:
    try:
        import dlimp as dl
        import tensorflow as tf

        from lerobot.datasets.rlds import dataset as dataset_module, oxe as oxe_module
        from lerobot.datasets.rlds.oxe import transforms as oxe_transforms_module
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
        make_dataset_from_webdataset=dataset_module.make_dataset_from_webdataset,
        make_restructure_fn=dataset_module._make_restructure_fn,
        apply_trajectory_transforms=dataset_module.apply_trajectory_transforms,
        apply_per_dataset_frame_transforms=dataset_module.apply_per_dataset_frame_transforms,
        apply_frame_transforms=dataset_module.apply_frame_transforms,
        get_oxe_dataset_kwargs_and_weights=oxe_module.get_oxe_dataset_kwargs_and_weights,
        named_mixtures=oxe_module.OXE_NAMED_MIXTURES,
        standardization_transforms=oxe_transforms_module.OXE_STANDARDIZATION_TRANSFORMS,
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


def _resolve_rlds_source_format(dataset_config: DatasetConfig, dataset_name: str) -> str:
    configured = dataset_config.rlds_storage_format
    has_tar_shards = bool(resolve_openx_tar_paths(dataset_config.root or "", dataset_name))
    if configured in {"auto", "hybrid"}:
        return "webdataset" if has_tar_shards else "tfds"
    if configured == "webdataset" and not has_tar_shards:
        raise FileNotFoundError(
            f"dataset.rlds_storage_format='webdataset', but no tar shards were found for "
            f"{dataset_name!r} under {dataset_config.root!r}."
        )
    return configured


def _adapt_openx_tar_source_kwargs(
    dataset_kwargs: Mapping[str, Any],
    root: str | Path,
    *,
    standardization_transforms: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    """Apply schema aliases specific to the local OpenX tar release."""
    source_kwargs = copy.deepcopy(dataset_kwargs)
    paths = resolve_openx_tar_paths(root, source_kwargs["name"])
    if source_kwargs["name"] == "bridge_orig" and paths and paths[0].parent.name == "bridge":
        # The tar release's ``bridge`` directory follows the Open-X schema,
        # whereas separately prepared bridge_orig TFDS data uses the legacy
        # flat-action schema. Replace both the semantic transform and the raw
        # observation keys; ActionMem loads images/state in addition to action.
        if standardization_transforms is None:
            raise ValueError("Bridge OpenX tar adaptation requires the OXE standardization registry.")
        source_kwargs["standardize_fn"] = standardization_transforms["bridge_oxe"]
        # The jxu124/OpenX tar export contains only the primary ``image``
        # camera for Bridge.  ``image_1`` belongs to the TFDS schema and is not
        # present in these pickled tar episodes, so represent every additional
        # requested view with the pipeline's normal empty-image padding.
        bridge_image_keys = {"primary": "image", "secondary": None, "wrist": None}
        source_kwargs["image_obs_keys"] = {
            view: bridge_image_keys[view] for view in source_kwargs.get("image_obs_keys", {})
        }
        if "state_obs_keys" in source_kwargs:
            source_kwargs["state_obs_keys"] = ["EEF_state", None, "gripper_state"]
        logging.info(
            "OpenX tar directory 'bridge' uses the bridge_oxe standardizer for source 'bridge_orig'."
        )
    return source_kwargs


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


def _attach_normalized_action_tokenizer_input(
    trajectory: dict[str, Any], statistics: Mapping[str, Any], tf: Any
):
    """Keep the per-source q01/q99-normalized action for action-code encoding.

    ``trajectory["action"]`` remains in the canonical, unnormalized ActionMem
    action space for metadata and environment post-processing. The flow target
    is normalized separately by the policy preprocessor.
    """
    action = tf.cast(trajectory["action"], tf.float32)
    action_stats = statistics["action"]
    low = tf.convert_to_tensor(action_stats["q01"], dtype=tf.float32)
    high = tf.convert_to_tensor(action_stats["q99"], dtype=tf.float32)
    mask = tf.convert_to_tensor(
        action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool)), dtype=tf.bool
    )
    normalized_action = tf.clip_by_value(2.0 * (action - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    normalized_action = tf.where(mask, normalized_action, action)

    # Match the vendored OXE BOUNDS_Q99 behavior for constant dimensions.
    minimum = tf.convert_to_tensor(action_stats["min"], dtype=tf.float32)
    maximum = tf.convert_to_tensor(action_stats["max"], dtype=tf.float32)
    normalized_action = tf.where(tf.equal(minimum, maximum), tf.zeros_like(action), normalized_action)
    trajectory[ACTION_TOKENIZER_INPUT] = normalized_action
    return trajectory


def _filter_vqvla_action_chunk(frame: Mapping[str, Any], chunk_filter_fn: Callable) -> Any:
    """Run VQ-VLA's source filter against its expected BOUNDS_Q99 action."""
    filter_frame = dict(frame)
    filter_frame["action"] = frame[ACTION_TOKENIZER_INPUT]
    return chunk_filter_fn(filter_frame)


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
            f"ActionMem action tokenizer expects {action_dim}."
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
    result = {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "std": torch.from_numpy(np.sqrt(variance).astype(np.float32)),
        "min": torch.from_numpy(minimum.astype(np.float32)),
        "max": torch.from_numpy(maximum.astype(np.float32)),
        "q01": torch.from_numpy(q01.astype(np.float32)),
        "q99": torch.from_numpy(q99.astype(np.float32)),
        "count": torch.tensor([count], dtype=torch.long),
    }
    source_masks = [stats[feature].get("mask") for stats in statistics]
    if any(mask is not None for mask in source_masks):
        padded_masks = []
        for stats, mask in zip(statistics, source_masks, strict=True):
            source_size = np.asarray(stats[feature]["mean"]).size
            source_mask = (
                np.ones(source_size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
            )
            if source_mask.size != source_size:
                raise ValueError(
                    f"RLDS {feature} normalization mask has size {source_mask.size}, expected {source_size}."
                )
            padded_masks.append(np.pad(source_mask, (0, size - source_size), constant_values=False))
        stacked_masks = np.stack(padded_masks)
        if feature == "action" and not np.all(stacked_masks == stacked_masks[0]):
            raise ValueError("All mixed OXE datasets must use the same action normalization mask.")
        result["mask"] = torch.from_numpy(np.all(stacked_masks, axis=0))
    return result


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
        action_tokenizer_checkpoint_path: str | Path,
        action_codebook_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset_config = dataset_config
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.seed = seed
        self.backend = _load_rlds_backend()
        self.rank, self.world_size = _resolve_rank_and_world_size()
        self._overfit_global_num_samples = dataset_config.rlds_overfit_num_samples
        self._overfit_num_samples: int | None = None
        self._overfit_samples: tuple[dict[str, Any], ...] | None = None
        if self._overfit_global_num_samples is not None:
            if self._overfit_global_num_samples < self.world_size:
                raise ValueError(
                    "rlds_overfit_num_samples must be at least the distributed world size "
                    f"({self.world_size}), got {self._overfit_global_num_samples}."
                )
            base_samples, remainder = divmod(self._overfit_global_num_samples, self.world_size)
            self._overfit_num_samples = base_samples + int(self.rank < remainder)
        self.backend.tf.random.set_seed(seed)
        self._webdataset_sources: list[dict[str, Any]] = []
        self._webdataset_sample_weights = np.empty(0, dtype=np.float64)
        self._hybrid_backend_weights = np.empty(0, dtype=np.float64)
        self._stream_shuffle_buffer_size = dataset_config.rlds_shuffle_buffer_size
        if self._overfit_num_samples is not None:
            self._stream_shuffle_buffer_size = min(
                self._stream_shuffle_buffer_size,
                self._overfit_num_samples,
            )
        self._webdataset_shuffle_buffer_size = self._stream_shuffle_buffer_size
        self.target_control_hz = (
            float(dataset_config.rlds_target_control_hz)
            if dataset_config.rlds_target_control_hz > 0
            else None
        )

        mixture_spec = _load_mixture_spec(dataset_config, self.backend.named_mixtures)
        per_dataset_kwargs, base_weights = self.backend.get_oxe_dataset_kwargs_and_weights(
            Path(dataset_config.root),
            mixture_spec,
            load_camera_views=tuple(dataset_config.rlds_camera_views),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=self.backend.normalization_type.NORMAL,
            target_control_hz=self.target_control_hz,
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
        self._dataset, all_statistics, effective_weights, valid_sizes = self._build_interleaved_dataset(
            per_dataset_kwargs,
            np.asarray(base_weights, dtype=np.float64),
        )
        self.dataset_names = [kwargs["name"] for kwargs in per_dataset_kwargs]
        self.source_formats = {
            name: _resolve_rlds_source_format(dataset_config, name) for name in self.dataset_names
        }
        self.sample_weights = effective_weights
        self.num_frames = int(
            max(size / weight for size, weight in zip(valid_sizes, effective_weights, strict=True))
        )
        self.num_episodes = sum(int(stats.get("num_trajectories", 0)) for stats in all_statistics)
        if self._overfit_global_num_samples is not None:
            self.num_frames = self._overfit_global_num_samples
            # MetricsTracker needs a non-zero episode count. Treat one pass
            # over the fixed debug cache as one synthetic episode/epoch.
            self.num_episodes = 1
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
            fps=self.target_control_hz or 1.0,
        )
        self.collate_fn = RLDSActionTokenCollator(
            checkpoint_path=action_tokenizer_checkpoint_path,
            device=dataset_config.rlds_action_tokenizer_device,
            expected_horizon=action_horizon,
            expected_action_dim=action_dim,
            expected_codebook_size=action_codebook_size,
            expected_target_control_hz=self.target_control_hz,
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
        logging.info(
            "RLDS storage backends: %s",
            ", ".join(f"{name}={self.source_formats[name]}" for name in self.dataset_names),
        )
        if self.target_control_hz is not None:
            logging.info(
                "OXE control-frequency alignment: %s",
                ", ".join(
                    f"{kwargs['name']}={kwargs['source_control_hz']:g}Hz->{self.target_control_hz:g}Hz"
                    for kwargs in per_dataset_kwargs
                ),
            )
        else:
            logging.info("OXE control-frequency alignment is disabled; using native source rates.")
        if self._overfit_num_samples is not None:
            logging.warning(
                "RLDS overfit mode is enabled: rank %d will cache %d fixed samples "
                "(%d globally) and repeat only those samples.",
                self.rank,
                self._overfit_num_samples,
                self._overfit_global_num_samples,
            )

    def _build_interleaved_dataset(
        self,
        per_dataset_kwargs: list[dict[str, Any]],
        base_weights: np.ndarray,
    ) -> tuple[Any, list[Mapping[str, Any]], np.ndarray, list[int]]:
        source_formats = [
            _resolve_rlds_source_format(self.dataset_config, kwargs["name"]) for kwargs in per_dataset_kwargs
        ]
        webdataset_indices = [
            index for index, source_format in enumerate(source_formats) if source_format == "webdataset"
        ]
        tfds_indices = [index for index, source_format in enumerate(source_formats) if source_format == "tfds"]

        all_statistics: list[Mapping[str, Any]] = []
        valid_sizes: list[int] = []
        for kwargs, source_format in zip(per_dataset_kwargs, source_formats, strict=True):
            statistics_kwargs = copy.deepcopy(kwargs)
            statistics_kwargs.pop("dataset_frame_transform_kwargs", None)
            if source_format == "webdataset":
                statistics_kwargs = _adapt_openx_tar_source_kwargs(
                    statistics_kwargs,
                    self.dataset_config.root or "",
                    standardization_transforms=self.backend.standardization_transforms,
                )
            make_source = (
                self.backend.make_dataset_from_webdataset
                if source_format == "webdataset"
                else self.backend.make_dataset_from_rlds
            )
            source_options = (
                {"seed": self.seed, "statistics_only": True} if source_format == "webdataset" else {}
            )
            _, statistics = make_source(
                **statistics_kwargs,
                train=True,
                num_parallel_calls=self.dataset_config.rlds_num_parallel_calls,
                num_parallel_reads=self.dataset_config.rlds_num_parallel_calls,
                **source_options,
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

        webdataset_weight = float(effective_weights[webdataset_indices].sum())
        tfds_weight = float(effective_weights[tfds_indices].sum())

        if webdataset_indices:
            self._webdataset_sample_weights = (
                effective_weights[webdataset_indices] / webdataset_weight
            )
            for index in webdataset_indices:
                source_kwargs = _adapt_openx_tar_source_kwargs(
                    per_dataset_kwargs[index],
                    self.dataset_config.root or "",
                    standardization_transforms=self.backend.standardization_transforms,
                )
                frame_transform_kwargs = source_kwargs.pop("dataset_frame_transform_kwargs", {})
                restructure = self.backend.make_restructure_fn(
                    name=source_kwargs["name"],
                    standardize_fn=source_kwargs.get("standardize_fn"),
                    image_obs_keys=source_kwargs.get("image_obs_keys", {}),
                    depth_obs_keys=source_kwargs.get("depth_obs_keys", {}),
                    state_obs_keys=source_kwargs.get("state_obs_keys", ()),
                    language_key=source_kwargs.get("language_key"),
                    absolute_action_mask=source_kwargs.get("absolute_action_mask"),
                    source_control_hz=source_kwargs.get("source_control_hz"),
                    target_control_hz=source_kwargs.get("target_control_hz"),
                )
                paths = resolve_openx_tar_paths(self.dataset_config.root or "", source_kwargs["name"])
                self._webdataset_sources.append(
                    {
                        "name": source_kwargs["name"],
                        "paths": paths,
                        "statistics": all_statistics[index],
                        "restructure": restructure,
                        "chunk_filter_fn": frame_transform_kwargs.get("chunk_filter_fn"),
                    }
                )

        using_hybrid_backends = bool(webdataset_indices and tfds_indices)
        if using_hybrid_backends:
            self._hybrid_backend_weights = np.asarray(
                [webdataset_weight, tfds_weight], dtype=np.float64
            )
            webdataset_buffer_size = int(round(self._stream_shuffle_buffer_size * webdataset_weight))
            self._webdataset_shuffle_buffer_size = max(1, webdataset_buffer_size)
            tfds_shuffle_buffer_size = max(
                1,
                self._stream_shuffle_buffer_size - self._webdataset_shuffle_buffer_size,
            )
        else:
            tfds_shuffle_buffer_size = self._stream_shuffle_buffer_size

        if not tfds_indices:
            return None, all_statistics, effective_weights, valid_sizes

        datasets = []
        tfds_sample_weights = effective_weights[tfds_indices] / tfds_weight
        threads_per_dataset = max(
            1, self.dataset_config.rlds_num_parallel_calls // len(tfds_indices)
        )
        for index in tfds_indices:
            source_kwargs = copy.deepcopy(per_dataset_kwargs[index])
            statistics = all_statistics[index]
            frame_transform_kwargs = source_kwargs.pop("dataset_frame_transform_kwargs", {})
            chunk_filter_fn = frame_transform_kwargs.pop("chunk_filter_fn", None)
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
                partial(
                    _attach_normalized_action_tokenizer_input,
                    statistics=statistics,
                    tf=self.backend.tf,
                ),
                threads_per_dataset,
            )
            source = source.traj_map(
                partial(_trim_invalid_chunk_targets, horizon=self.action_horizon, tf=self.backend.tf),
                threads_per_dataset,
            ).flatten(num_parallel_calls=threads_per_dataset)
            if chunk_filter_fn is not None:
                source = source.filter(partial(_filter_vqvla_action_chunk, chunk_filter_fn=chunk_filter_fn))
            source = self.backend.apply_per_dataset_frame_transforms(source, **frame_transform_kwargs)
            datasets.append(source)

        mixed = self.backend.dl.DLataset.sample_from_datasets(datasets, tfds_sample_weights)
        mixed = mixed.shuffle(tfds_shuffle_buffer_size)
        mixed = self.backend.apply_frame_transforms(
            mixed,
            train=True,
            resize_size=tuple(self.dataset_config.rlds_resize_size),
            num_parallel_calls=self.dataset_config.rlds_num_parallel_calls,
        )
        return mixed.with_ram_budget(1), all_statistics, effective_weights, valid_sizes

    def _decode_webdataset_image(self, encoded: Any) -> np.ndarray:
        encoded_array = np.asarray(encoded)
        is_decoded_image = encoded_array.ndim >= 2 and encoded_array.dtype.kind not in {"O", "S", "U"}
        if is_decoded_image:
            image = Image.fromarray(encoded_array.astype(np.uint8, copy=False))
        else:
            if encoded_array.ndim == 0:
                encoded = encoded_array.item()
            if not encoded:
                return np.zeros((*self.dataset_config.rlds_resize_size, 3), dtype=np.uint8)
            image = Image.open(io.BytesIO(encoded))
        height, width = self.dataset_config.rlds_resize_size
        with image:
            image = image.convert("RGB")
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
            return np.asarray(image, dtype=np.uint8)

    def _iter_webdataset_source(self, source: Mapping[str, Any], source_index: int):
        statistics = source["statistics"]
        action_stats = statistics["action"]
        q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        minimum = np.asarray(action_stats["min"], dtype=np.float32)
        maximum = np.asarray(action_stats["max"], dtype=np.float32)
        normalization_mask = np.asarray(
            action_stats.get("mask", np.ones(self.action_dim, dtype=bool)),
            dtype=bool,
        )
        chunk_filter_fn = source.get("chunk_filter_fn")
        epoch = 0
        while True:
            yielded_in_epoch = False
            episode_stream = iter_openx_tar_episodes(
                source["paths"],
                seed=self.seed + epoch,
                shuffle_shards=True,
            )
            frame_rng = np.random.default_rng(self.seed + 10_007 * source_index + epoch)
            for episode_index, payload in enumerate(episode_stream):
                if episode_index % self.world_size != self.rank:
                    continue
                trajectory = transform_openx_tar_episode(
                    payload,
                    tf=self.backend.tf,
                    transform=source["restructure"],
                )
                action = np.asarray(trajectory["action"], dtype=np.float32)
                observation = trajectory["observation"]
                proprio = np.asarray(observation["proprio"], dtype=np.float32)
                language = np.asarray(trajectory["task"]["language_instruction"])
                if self.dataset_config.rlds_skip_unlabeled and not any(
                    _decode_text(value) for value in language
                ):
                    continue
                valid_length = action.shape[0] - self.action_horizon + 1
                if valid_length <= 0:
                    continue
                frame_indices = np.arange(valid_length)
                frame_rng.shuffle(frame_indices)
                for frame_index in frame_indices:
                    action_chunk = action[frame_index : frame_index + self.action_horizon]
                    normalized_chunk = np.clip(
                        2.0 * (action_chunk - q01) / (q99 - q01 + 1e-8) - 1.0,
                        -1.0,
                        1.0,
                    )
                    normalized_chunk = np.where(normalization_mask, normalized_chunk, action_chunk)
                    normalized_chunk = np.where(minimum == maximum, 0.0, normalized_chunk).astype(np.float32)
                    if chunk_filter_fn is not None:
                        keep = chunk_filter_fn(
                            {"action": self.backend.tf.convert_to_tensor(normalized_chunk)}
                        )
                        if not bool(np.asarray(keep)):
                            continue

                    pad_mask_dict: dict[str, np.ndarray] = {}
                    frame_observation: dict[str, Any] = {
                        "proprio": proprio[frame_index : frame_index + 1],
                        "pad_mask_dict": pad_mask_dict,
                    }
                    for view in self.dataset_config.rlds_camera_views:
                        key = f"image_{view}"
                        encoded = np.asarray(observation[key])[frame_index]
                        encoded_array = np.asarray(encoded)
                        image_is_present = (
                            True
                            if encoded_array.ndim >= 2
                            else bool(encoded_array.item() if encoded_array.ndim == 0 else encoded)
                        )
                        # Keep images compressed while frames are in the large
                        # cross-episode shuffle buffer. Decode only the frame
                        # selected for the next training sample.
                        frame_observation[key] = encoded
                        pad_mask_dict[key] = np.asarray([image_is_present], dtype=bool)

                    yielded_in_epoch = True
                    yield {
                        "observation": frame_observation,
                        "task": {"language_instruction": language[frame_index]},
                        "action": action_chunk,
                        ACTION_TOKENIZER_INPUT: normalized_chunk,
                        "dataset_name": source["name"],
                    }
            if not yielded_in_epoch:
                raise ValueError(
                    f"OpenX tar source {source['name']!r} produced no valid samples on rank "
                    f"{self.rank}/{self.world_size}."
                )
            epoch += 1

    def _materialize_webdataset_frame(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        materialized = dict(frame)
        observation = dict(frame["observation"])
        materialized["observation"] = observation
        for view in self.dataset_config.rlds_camera_views:
            key = f"image_{view}"
            observation[key] = self._decode_webdataset_image(observation[key])[None]
        return materialized

    def _iter_weighted_webdatasets(self):
        iterators = [
            iter(self._iter_webdataset_source(source, source_index))
            for source_index, source in enumerate(self._webdataset_sources)
        ]
        rng = np.random.default_rng(self.seed + self.rank)
        shuffle_buffer: list[Mapping[str, Any]] = []

        def next_weighted_frame():
            source_index = int(rng.choice(len(iterators), p=self._webdataset_sample_weights))
            return next(iterators[source_index])

        logging.info(
            "Filling the OpenX tar shuffle buffer with %d compressed frames before training starts.",
            self._webdataset_shuffle_buffer_size,
        )
        while len(shuffle_buffer) < self._webdataset_shuffle_buffer_size:
            shuffle_buffer.append(next_weighted_frame())
            if len(shuffle_buffer) % 10_000 == 0:
                logging.info(
                    "OpenX tar shuffle buffer: %d/%d frames.",
                    len(shuffle_buffer),
                    self._webdataset_shuffle_buffer_size,
                )
        logging.info("OpenX tar shuffle buffer is ready.")
        if self._overfit_num_samples is not None:
            # Let fixed-cache mode see each initially buffered frame once
            # before normal replacement sampling can introduce duplicates.
            for buffer_index in rng.permutation(len(shuffle_buffer)):
                yield self._materialize_webdataset_frame(shuffle_buffer[int(buffer_index)])
        while True:
            buffer_index = int(rng.integers(len(shuffle_buffer)))
            frame = shuffle_buffer[buffer_index]
            shuffle_buffer[buffer_index] = next_weighted_frame()
            yield self._materialize_webdataset_frame(frame)

    def _iter_tfds_frames(self):
        if self._dataset is None:
            raise RuntimeError("The TFDS/RLDS stream has not been initialized.")
        while True:
            yielded = False
            for frame in self._dataset.as_numpy_iterator():
                yielded = True
                yield frame
            if not yielded:
                raise ValueError("The TFDS/RLDS stream is empty.")

    def _iter_hybrid_frames(self):
        backend_iterators = [
            iter(self._iter_weighted_webdatasets()),
            iter(self._iter_tfds_frames()),
        ]
        rng = np.random.default_rng(self.seed + 47_021 + self.rank)
        while True:
            backend_index = int(rng.choice(2, p=self._hybrid_backend_weights))
            yield next(backend_iterators[backend_index])

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
        action_tokenizer_input = np.asarray(frame[ACTION_TOKENIZER_INPUT], dtype=np.float32)
        if action_tokenizer_input.shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                "Expected q01/q99-normalized RLDS action-tokenizer chunk "
                f"{(self.action_horizon, self.action_dim)}, got {action_tokenizer_input.shape}."
            )
        sample: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(state),
            ACTION: torch.from_numpy(action.copy()),
            ACTION_TOKENIZER_INPUT: torch.from_numpy(action_tokenizer_input.copy()),
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

    def _iter_source_frames(self):
        if self._webdataset_sources and self._dataset is not None:
            return self._iter_hybrid_frames()
        if self._webdataset_sources:
            return self._iter_weighted_webdatasets()
        return self._iter_tfds_frames()

    def __iter__(self):
        frames = self._iter_source_frames()
        if self._overfit_num_samples is None:
            for frame in frames:
                yield self._to_lerobot_sample(frame)
            return

        if self._overfit_samples is None:
            cached_samples: list[dict[str, Any]] = []
            for frame in frames:
                cached_samples.append(self._to_lerobot_sample(frame))
                if len(cached_samples) == self._overfit_num_samples:
                    break
            if len(cached_samples) != self._overfit_num_samples:
                raise RuntimeError(
                    f"RLDS overfit cache expected {self._overfit_num_samples} samples on rank "
                    f"{self.rank}, but collected {len(cached_samples)}."
                )
            self._overfit_samples = tuple(cached_samples)
            logging.warning(
                "RLDS overfit cache is ready on rank %d: %d fixed samples.",
                self.rank,
                len(self._overfit_samples),
            )

        epoch = 0
        while True:
            order = np.random.default_rng(self.seed + 104_729 * self.rank + epoch).permutation(
                len(self._overfit_samples)
            )
            for sample_index in order:
                yield self._overfit_samples[int(sample_index)]
            epoch += 1

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int):
        raise NotImplementedError("ActionMemRLDSDataset is an infinite IterableDataset.")


def make_actionmem_rlds_dataset(cfg: Any) -> ActionMemRLDSDataset:
    policy_config = cfg.trainable_config
    checkpoint_path = (
        cfg.dataset.rlds_effect_tokenizer_checkpoint_path
        or policy_config.effect_tokenizer_checkpoint_path
    )
    if checkpoint_path is None:
        raise ValueError(
            f"{policy_config.type} RLDS training requires an effect-tokenizer checkpoint. Set "
            "--dataset.rlds_effect_tokenizer_checkpoint_path or "
            "--policy.effect_tokenizer_checkpoint_path."
        )
    metadata = load_effect_tokenizer_metadata(checkpoint_path)
    action_horizon = metadata.horizon
    action_dim = metadata.action_dim
    requested_hz = (
        float(cfg.dataset.rlds_target_control_hz)
        if cfg.dataset.rlds_target_control_hz > 0
        else None
    )
    frequency_mismatch = (metadata.target_control_hz is None) != (requested_hz is None)
    if metadata.target_control_hz is not None and requested_hz is not None:
        frequency_mismatch = not np.isclose(metadata.target_control_hz, requested_hz)
    if frequency_mismatch:
        raise ValueError(
            f"dataset.rlds_target_control_hz={requested_hz} does not match effect-tokenizer "
            f"target_control_hz={metadata.target_control_hz}."
        )
    action_codebook_size = int(policy_config.action_codebook_size)
    if action_codebook_size != metadata.codebook_size:
        raise ValueError(
            f"Policy action_codebook_size={action_codebook_size} does not match effect-tokenizer "
            f"codebook_size={metadata.codebook_size}."
        )

    configured_chunk_size = int(getattr(cfg.trainable_config, "chunk_size", action_horizon))
    if configured_chunk_size != action_horizon:
        raise ValueError(
            f"Policy chunk_size={configured_chunk_size} must match action-tokenizer horizon={action_horizon}."
        )
    state_dim = cfg.dataset.rlds_state_dim or int(getattr(cfg.trainable_config, "max_state_dim", 32))
    return ActionMemRLDSDataset(
        dataset_config=cfg.dataset,
        action_horizon=action_horizon,
        action_dim=action_dim,
        state_dim=state_dim,
        action_tokenizer_checkpoint_path=checkpoint_path,
        action_codebook_size=action_codebook_size,
        seed=cfg.seed if cfg.seed is not None else 0,
    )
