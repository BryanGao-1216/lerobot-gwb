#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import draccus
from huggingface_hub import hf_hub_download

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224
LAM_IMAGE_PAIR = "observation.pi05_lap_image_pair"


@PreTrainedConfig.register_subclass("pi05_lap")
@dataclass
class PI05LAPConfig(PreTrainedConfig):
    training_stage: str = "post-training"
    lam_model_path: str | None = None
    # None selects the primary RGB view from dataset metadata; accepts one short or full key.
    lam_camera_key: str | None = None

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.training_stage not in {"pre-training", "post-training"}:
            raise ValueError("training_stage must be 'pre-training' or 'post-training'.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if self.lam_camera_key is not None and (
            not isinstance(self.lam_camera_key, str)
            or not self.lam_camera_key.strip()
            or any(c in self.lam_camera_key for c in ",[]")
        ):
            raise ValueError("lam_camera_key must name exactly one RGB camera.")
        if self.training_stage == "pre-training" and self.train_expert_only:
            raise ValueError("VLM pre-training requires train_expert_only=false.")

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def resolve_lam_camera_key(self, features: dict) -> str:
        cameras = []
        for key, feature in features.items():
            if not key.startswith(OBS_IMAGES + ".") or feature.get("dtype") not in {"image", "video"}:
                continue
            shape = feature.get("shape", ())
            names = feature.get("names") or ()
            if len(shape) == 3 and "depth" not in key.lower():
                channel_axis = names.index("channels") if "channels" in names else None
                if (channel_axis is not None and shape[channel_axis] == 3) or (
                    channel_axis is None and (shape[0] == 3 or shape[-1] == 3)
                ):
                    cameras.append(key)
        if self.lam_camera_key is not None:
            camera = self.lam_camera_key.strip()
            if not camera.startswith(OBS_IMAGES + "."):
                camera = f"{OBS_IMAGES}.{camera}"
            if camera not in cameras:
                raise ValueError(f"LAM camera {camera!r} is not RGB. Available RGB cameras: {cameras}")
        else:
            preferred = ("image", "base_0_rgb", "primary", "front", "agentview_image", "image_primary")
            camera = next(
                (f"{OBS_IMAGES}.{name}" for name in preferred if f"{OBS_IMAGES}.{name}" in cameras), None
            )
            if camera is None:
                if not cameras:
                    raise ValueError("UniVLA pre-training needs an RGB camera in the dataset.")
                camera = cameras[0]
        self.lam_camera_key = camera
        return camera

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, cli_overrides=None, **kwargs):
        """Allow an original PI05 config to initialize the new policy without losing its settings."""
        path = Path(pretrained_name_or_path)
        if path.is_dir():
            config_file = path / "config.json"
        else:
            hub_args = {
                k: v
                for k, v in kwargs.items()
                if k in {"force_download", "token", "cache_dir", "local_files_only", "revision"}
            }
            config_file = hf_hub_download(str(pretrained_name_or_path), "config.json", **hub_args)
        with open(config_file) as f:
            source = json.load(f)
        source_type = source.pop("type", None)
        if source_type not in {"pi05", "pi05_lap"}:
            raise ValueError(f"pi05_lap requires a PI05 or PI05 LAP checkpoint, got {source_type!r}.")
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"
            config_file.write_text(json.dumps(source))
            with draccus.config_type("json"):
                return draccus.parse(cls, config_file, args=cli_overrides or [])

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
