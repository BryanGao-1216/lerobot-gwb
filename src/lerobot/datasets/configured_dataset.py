"""Map heterogeneous LeRobot sources to a shared PI05 training schema.

Each child owns its episode-clamped temporal queries. The outer dataset only
maps fields, pads dimensions/views and exposes a deterministic weighted sampler.
"""

import logging
import math
from bisect import bisect_right
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.configs.dataset_mixture import DatasetSourceConfig, VectorFieldConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_STATE

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .sampler import EpisodeAwareSampler, WeightedDatasetSampler

_NEUTRAL_VECTOR_STATS = {"mean": 0, "std": 1, "min": -1, "max": 1, "q01": -1, "q99": 1, "q10": -1, "q90": 1}


def _vector_indices(fields: list[VectorFieldConfig], features: dict) -> list[tuple[str, list[int]]]:
    result = []
    for spec in fields:
        if spec.key not in features:
            raise ValueError(f"Missing vector field {spec.key!r}. Available: {list(features)}")
        shape = features[spec.key]["shape"]
        if len(shape) != 1 or features[spec.key]["dtype"] in {"image", "video", "string"}:
            raise ValueError(f"{spec.key} must be a numeric vector, got {features[spec.key]}.")
        indices = spec.indices if spec.indices is not None else list(range(shape[0]))
        if not indices or max(indices) >= shape[0]:
            raise ValueError(f"{spec.key}: selected indices exceed vector shape {shape}.")
        result.append((spec.key, indices))
    return result


def _vector_stats(meta, fields, size: int, mode: NormalizationMode) -> dict:
    required = {
        NormalizationMode.IDENTITY: (),
        NormalizationMode.MEAN_STD: ("mean", "std"),
        NormalizationMode.MIN_MAX: ("min", "max"),
        NormalizationMode.QUANTILES: ("q01", "q99"),
        NormalizationMode.QUANTILE10: ("q10", "q90"),
    }[mode]
    # Neutral statistics keep padded zero components exactly zero after normalization.
    result = {"count": torch.tensor([meta.total_frames])}
    for stat in required:
        values = []
        for key, indices in fields:
            if stat not in (meta.stats or {}).get(key, {}):
                raise ValueError(f"{meta.repo_id}: missing {key}.{stat} normalization statistics.")
            vector = torch.as_tensor(meta.stats[key][stat], dtype=torch.float32).flatten()
            if vector.numel() != meta.features[key]["shape"][0]:
                raise ValueError(f"{meta.repo_id}: {key}.{stat} shape differs from the feature shape.")
            values.append(vector[indices])
        combined = torch.cat(values)
        if not torch.isfinite(combined).all():
            raise ValueError(f"{meta.repo_id}: non-finite {stat} normalization statistics.")
        result[stat] = F.pad(combined, (0, size - len(combined)), value=_NEUTRAL_VECTOR_STATS[stat])
    return result


def _gripper_indices(source, meta, action_fields) -> list[int]:
    if source.gripper_mapping is None:
        return []
    action_dim = sum(len(indices) for _, indices in action_fields)
    if action_dim in (7, 14):
        return list(range(6, action_dim, 7))
    # Also support explicitly named gripper fields in other vector layouts.
    result, offset = [], 0
    for key, indices in action_fields:
        names = meta.features[key].get("names")
        if names and isinstance(names[0], list):
            names = names[0]
        for index in indices:
            name = names[index] if names and len(names) == meta.features[key]["shape"][0] else key
            if "gripper" in key.lower() or "gripper" in str(name).lower():
                result.append(offset)
            offset += 1
    if not result:
        raise ValueError(
            f"{source.repo_id}: discrete gripper mapping requires 7/14-D actions "
            "or action fields/components named 'gripper'."
        )
    return result


def _map_discrete_gripper(action, source, indices):
    mapping = source.gripper_mapping
    if mapping is None:
        return action
    values = action[..., indices]
    endpoints = values.new_tensor([mapping.open, mapping.close])
    is_open = torch.isclose(values, endpoints[0], rtol=0, atol=1e-6)
    is_closed = torch.isclose(values, endpoints[1], rtol=0, atol=1e-6)
    if (is_open & is_closed).any() or not (is_open | is_closed).all():
        raise ValueError(
            f"{source.repo_id}: discrete gripper labels must match open={mapping.open}, "
            f"close={mapping.close}; found {values.unique().tolist()}."
        )
    action[..., indices] = torch.where(is_open, -1.0, 1.0)
    return action


def _relative_action_mask(source, meta, action_fields, state_fields, size, gripper_indices=()):
    mask = []
    excluded = [name.lower() for name in source.relative_exclude_joints if name]
    for key, indices in action_fields:
        names = meta.features[key].get("names")
        if names and isinstance(names[0], list):
            names = names[0]
        for index in indices:
            name = names[index] if names is not None and len(names) == meta.features[key]["shape"][0] else key
            mask.append(source.use_relative_actions and not any(token in name.lower() for token in excluded))
    # Canonical discrete commands must not be changed by action-state conversion.
    for index in gripper_indices:
        mask[index] = False
    state_dim = sum(len(indices) for _, indices in state_fields)
    if any(mask[state_dim:]):
        raise ValueError(
            f"{source.repo_id}: relative action dimensions need matching state components; "
            "align action_fields/state_fields or exclude the unmatched joints."
        )
    return mask + [False] * (size - len(mask))


def _split_episodes(meta, source: DatasetSourceConfig, default_split: float):
    episodes = source.episodes if source.episodes is not None else list(range(meta.total_episodes))
    if any(ep >= meta.total_episodes for ep in episodes):
        raise ValueError(f"{source.repo_id}: episode selection is out of range.")
    split = source.eval_split if source.eval_split is not None else default_split
    if split == 0:
        return sorted(episodes), []
    by_task = {}
    task_keys = meta.get_episode_task_keys(episodes)
    for ep in sorted(episodes):
        by_task.setdefault(task_keys[ep], []).append(ep)
    train, evaluation = [], []
    for group in by_task.values():
        n = math.ceil(len(group) * split)
        train.extend(group[:-n])
        evaluation.extend(group[-n:])
    if not train:
        raise ValueError(f"{source.repo_id}: eval_split={split} leaves no training episodes.")
    return sorted(train), sorted(evaluation)


def _validate_rgb_camera(meta, key: str, source_name: str) -> None:
    feature = meta.features.get(key, {})
    shape = feature.get("shape", ())
    names = feature.get("names") or []
    axis = names.index("channels") if "channels" in names else None
    rgb = len(shape) == 3 and (shape[axis] == 3 if axis is not None else 3 in (shape[0], shape[-1]))
    if key not in meta.camera_keys or key in meta.depth_keys or not rgb:
        raise ValueError(f"{source_name}: camera {key!r} must be an RGB image/video field.")


class ConfiguredDataset(torch.utils.data.Dataset):
    def __init__(self, children, sources, schema, *, image_transforms=None, is_lap=False):
        self.children = children
        self.sources = sources
        self.schema = schema
        self.image_transforms = image_transforms
        self.is_lap = is_lap
        self._lengths = [len(index) for _, _, index in children]
        self._ends = np.cumsum(self._lengths).tolist()
        self.num_frames = sum(self._lengths)
        self.num_episodes = sum(len(index._starts) for _, _, index in children)
        self.episodes = None
        self.absolute_to_relative_idx = None
        self.meta = SimpleNamespace(
            features=schema.features,
            stats=schema.stats,
            camera_keys=schema.camera_keys,
            depth_keys=[],
            has_language_columns=False,
            total_frames=self.num_frames,
            total_episodes=self.num_episodes,
            repo_id=[source.repo_id for source in sources.values()],
            robot_type="mixed",
            fps=schema.fps,
            tasks=schema.tasks,
            is_mixture=True,
            relative_action_masks=schema.relative_action_masks,
        )

    def __len__(self):
        return self.num_frames

    def make_sampler(self, seed: int, batch_multiple: int = 1):
        return WeightedDatasetSampler(
            self._lengths,
            [self.sources[name].weight for name, _, _ in self.children],
            seed=seed,
            batch_multiple=batch_multiple,
        )

    def eval_indices(self, max_samples: int) -> list[int]:
        """Cap offline evaluation across tasks using episode metadata, without decoding frames."""
        by_task = {}
        offset = 0
        for _, child, index in self.children:
            # Locate the original episodes containing these intervals.
            episodes = [
                bisect_right(child.meta.episodes["dataset_from_index"], int(start)) - 1
                for start in index._starts
            ]
            task_keys = child.meta.get_episode_task_keys(episodes)
            previous = 0
            for episode, end in zip(episodes, index._cum_lengths, strict=True):
                task = task_keys[episode]
                by_task.setdefault(task, []).append(range(offset + previous, offset + int(end)))
                previous = int(end)
            offset += len(index)
        quota, remainder = divmod(max_samples, len(by_task))
        selected = []
        for i, ranges in enumerate(by_task.values()):
            remaining = quota + (i < remainder)
            for interval in ranges:
                take = min(remaining, len(interval))
                selected.extend(interval[:take])
                remaining -= take
                if remaining == 0:
                    break
        return selected

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        child_index = bisect_right(self._ends, idx)
        name, dataset, frame_index = self.children[child_index]
        source = self.sources[name]
        position = idx - (self._ends[child_index - 1] if child_index else 0)
        raw = dataset[frame_index._frame_index(position)]
        source_index = self.schema.source_indices[name]
        item = {
            "dataset_index": torch.tensor(source_index),
            "image_augmentation": torch.tensor(source.image_augmentation),
            "index": torch.tensor(idx),
            "episode_index": raw["episode_index"] + self.schema.episode_offsets[name],
            "frame_index": raw["frame_index"],
            "timestamp": raw["timestamp"],
            "task": raw["task"],
            "task_index": torch.tensor(self.schema.task_indices[raw["task"]]),
        }
        for target, fields, size in (
            (ACTION, self.schema.action_fields[name], self.schema.action_dim),
            (OBS_STATE, self.schema.state_fields[name], self.schema.state_dim),
        ):
            values = []
            for key, indices in fields:
                value = raw[key].float()
                if tuple(dataset.meta.features[key]["shape"]) == (1,):
                    # HF scalarizes one-component numeric features on disk.
                    expected_ndim = 2 if key in dataset.reader.delta_indices else 1
                    if value.ndim < expected_ndim:
                        value = value.unsqueeze(-1)
                # A field used both as state and action was queried as a chunk; state stays at t.
                if target == OBS_STATE and value.ndim == 2:
                    value = value[0]
                values.append(value[..., indices])
            value = torch.cat(values, dim=-1)
            if target == ACTION:
                value = _map_discrete_gripper(value, source, self.schema.gripper_indices[name])
            item[target] = F.pad(value, (0, size - value.shape[-1]))
            item[f"{target}_dim_is_pad"] = torch.arange(size) >= value.shape[-1]
        item["action_is_pad"] = torch.stack(
            [raw[f"{key}_is_pad"] for key, _ in self.schema.action_fields[name]]
        ).any(dim=0)

        lam_camera = self.schema.lam_camera_keys.get(name)
        if self.is_lap:
            from lerobot.policies.pi05_lap.configuration_pi05_lap import LAM_IMAGE_PAIR
            from lerobot.policies.pi05_lap.lam.preprocessing import preprocess_lam_pair

            # Take the selected student's camera pair before resizing/augmentation.
            item[LAM_IMAGE_PAIR] = preprocess_lam_pair(raw[lam_camera], 224)
        for camera in self.schema.camera_keys:
            source_key = source.camera_fields.get(camera)
            if source_key is None:
                item[camera] = torch.zeros(
                    3,
                    *self.schema.image_resolution,
                    dtype=torch.uint8 if self.schema.return_uint8 else torch.float32,
                )
                item[f"{camera}_is_pad"] = torch.tensor(True)
                continue
            image = raw[source_key]
            if self.is_lap and source_key == lam_camera:
                image = image[0]
            # Image columns are decoded as float32; videos may be uint8. Do not
            # let default_collate promote byte pixels to unscaled floats in mixed batches.
            if self.schema.return_uint8 and image.dtype != torch.uint8:
                image = (image * 255).to(torch.uint8)
            elif not self.schema.return_uint8 and image.dtype == torch.uint8:
                image = image.float().div(255)
            # Resize before collating so different source resolutions can share a batch.
            from lerobot.policies.common.openpi_io import resize_openpi_image

            image = resize_openpi_image(image, *self.schema.image_resolution)
            if self.image_transforms is not None:
                image = self.image_transforms(image)
            item[camera] = image
            item[f"{camera}_is_pad"] = torch.tensor(False)
        return item


def make_configured_datasets(cfg):
    policy = cfg.policy
    settings = cfg.dataset
    sources = settings.datasets
    is_lap = policy.type == "pi05_lap" and policy.training_stage == "pre-training"
    metadata = {
        name: LeRobotDatasetMetadata(source.repo_id, root=source.root, revision=source.revision)
        for name, source in sources.items()
    }
    action_fields, state_fields, cameras, stats = {}, {}, [], {}
    gripper_indices = {}
    lam_camera_keys = {}
    action_dim, state_dim = 0, 0
    episode_offsets, episode_offset = {}, 0
    task_names = []
    for name, source in sources.items():
        meta = metadata[name]
        action_fields[name] = _vector_indices(source.action_fields, meta.features)
        state_fields[name] = _vector_indices(source.state_fields, meta.features)
        gripper_indices[name] = _gripper_indices(source, meta, action_fields[name])
        action_dim = max(action_dim, sum(len(i) for _, i in action_fields[name]))
        state_dim = max(state_dim, sum(len(i) for _, i in state_fields[name]))
        episode_offsets[name], episode_offset = episode_offset, episode_offset + meta.total_episodes
        task_names.extend(str(task) for task in meta.tasks.index if str(task) not in task_names)
        for target, key in source.camera_fields.items():
            if key is not None:
                _validate_rgb_camera(meta, key, name)
            if target not in cameras:
                cameras.append(target)
        if is_lap:
            lam_camera_keys[name] = source.resolve_lam_camera_key()
            _validate_rgb_camera(meta, lam_camera_keys[name], name)
    if len(cameras) > 3:
        raise ValueError("OpenPI PI05/LAP supports at most three shared camera slots across datasets.")
    if action_dim > policy.max_action_dim or state_dim > policy.max_state_dim:
        raise ValueError(
            f"Mapped action/state dimensions ({action_dim}, {state_dim}) exceed policy capacity "
            f"({policy.max_action_dim}, {policy.max_state_dim}). Select indices explicitly; no truncation is applied."
        )
    features = {
        ACTION: {"dtype": "float32", "shape": (action_dim,), "names": None},
        OBS_STATE: {"dtype": "float32", "shape": (state_dim,), "names": None},
        **{
            key: {
                "dtype": "image",
                "shape": (3, *policy.image_resolution),
                "names": ["channels", "height", "width"],
            }
            for key in cameras
        },
    }
    relative_action_masks = {}
    for source_index, (name, source) in enumerate(sources.items()):
        meta = metadata[name]
        relative_action_masks[str(source_index)] = _relative_action_mask(
            source, meta, action_fields[name], state_fields[name], action_dim, gripper_indices[name]
        )
        for key, fields, size, feature_type in (
            (ACTION, action_fields[name], action_dim, FeatureType.ACTION),
            (OBS_STATE, state_fields[name], state_dim, FeatureType.STATE),
        ):
            mode = NormalizationMode(policy.normalization_mapping.get(feature_type.value, "IDENTITY"))
            # Prefixes are understood by the regular normalizer and saved in its safetensors.
            stats[f"dataset_{source_index}/{key}"] = _vector_stats(meta, fields, size, mode)
            if key == ACTION and gripper_indices[name]:
                # Fixed canonical statistics keep all discrete sources in the same
                # model space, including one-state datasets and MEAN_STD normalization.
                for stat, vector in stats[f"dataset_{source_index}/{key}"].items():
                    if stat in _NEUTRAL_VECTOR_STATS:
                        vector[gripper_indices[name]] = _NEUTRAL_VECTOR_STATS[stat]
        for camera, key in source.camera_fields.items():
            if key is None:
                continue
            stats[f"dataset_{source_index}/{camera}"] = (
                IMAGENET_STATS if settings.use_imagenet_stats else (meta.stats or {}).get(key, {})
            )
    fps_values = {meta.fps for meta in metadata.values()}
    schema = SimpleNamespace(
        features=features,
        stats=stats,
        camera_keys=cameras,
        action_dim=action_dim,
        state_dim=state_dim,
        action_fields=action_fields,
        state_fields=state_fields,
        gripper_indices=gripper_indices,
        relative_action_masks=relative_action_masks,
        lam_camera_keys=lam_camera_keys,
        image_resolution=policy.image_resolution,
        source_indices={name: i for i, name in enumerate(sources)},
        episode_offsets=episode_offsets,
        fps=next(iter(fps_values)) if len(fps_values) == 1 else None,
        tasks=pd.DataFrame({"task_index": range(len(task_names))}, index=task_names),
        task_indices={name: i for i, name in enumerate(task_names)},
        return_uint8=settings.return_uint8,
    )
    train_children, eval_children = [], []
    for name, source in sources.items():
        meta = metadata[name]
        train_episodes, eval_episodes = _split_episodes(meta, source, settings.eval_split)
        deltas = {key: [i / meta.fps for i in policy.action_delta_indices] for key, _ in action_fields[name]}
        loaded_cameras = [key for key in source.camera_fields.values() if key is not None]
        if is_lap:
            camera = lam_camera_keys[name]
            deltas[camera] = [0, policy.chunk_size / meta.fps]
        for episodes, children in ((train_episodes, train_children), (eval_episodes, eval_children)):
            if not episodes:
                continue
            child = LeRobotDataset(
                source.repo_id,
                root=source.root,
                # Avoid an O(num_frames) absolute->relative dict when selecting the whole source.
                episodes=None if len(episodes) == meta.total_episodes else episodes,
                revision=source.revision,
                delta_timestamps=deltas,
                image_transforms=None,
                video_backend=source.video_backend or settings.video_backend,
                return_uint8=settings.return_uint8,
                depth_output_unit=settings.depth_output_unit,
                tolerance_s=source.tolerance_s or settings.tolerance_s or cfg.tolerance_s,
                camera_keys=loaded_cameras,
            )
            index = EpisodeAwareSampler(
                meta.episodes["dataset_from_index"],
                meta.episodes["dataset_to_index"],
                episode_indices_to_use=episodes,
                drop_n_last_frames=getattr(policy, "drop_n_last_frames", 0),
                absolute_to_relative_idx=child.absolute_to_relative_idx,
            )
            children.append((name, child, index))
        logging.info(
            "Dataset %s: weight=%s action_dim=%d state_dim=%d train_episodes=%d eval_episodes=%d",
            name,
            source.weight,
            sum(len(i) for _, i in action_fields[name]),
            sum(len(i) for _, i in state_fields[name]),
            len(train_episodes),
            len(eval_episodes),
        )
    transforms = ImageTransforms(settings.image_transforms) if settings.image_transforms.enable else None
    train = ConfiguredDataset(train_children, sources, schema, image_transforms=transforms, is_lap=is_lap)
    evaluation = ConfiguredDataset(eval_children, sources, schema, is_lap=is_lap) if eval_children else None
    return train, evaluation
