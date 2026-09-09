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

"""Configuration for SmolW, a joint VidTwin-z/action flow policy."""

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolw")
@dataclass
class SmolWConfig(SmolVLAConfig):
    """SmolVLA jointly generating future VidTwin z and an action chunk.

    The VLM still receives only the current observation. Temporal frames are
    requested from a LeRobot dataset through ``observation_delta_indices`` and
    are consumed by the frozen VidTwin motion extractor.
    """

    # None preserves checkpoints written before the controlled A/B/C/D ablations.
    # Launch scripts must explicitly choose A, B, C, or D for new experiments.
    train_version: str | None = None

    # Motion/action temporal contract. ``None`` keeps the motion horizon tied to
    # the SmolVLA action chunk size.
    motion_horizon: int | None = None
    memory_stride: int = 1

    # SmolW always trains the VLM condition path and action expert jointly.
    train_expert_only: bool = False

    # The frozen VidTwin extractor is external to the policy state dict. Its
    # architecture is bundled with SmolW; only checkpoint weights are external.
    vidtwin_checkpoint_path: str | None = None
    vidtwin_num_frames: int = 16
    vidtwin_input_height: int = 224
    vidtwin_input_width: int = 224
    vidtwin_dtype: str = "bfloat16"
    # Deterministic posterior modes are substantially better regression targets
    # than resampling the Gaussian posterior on every training step.
    vidtwin_sample_posterior: bool = False

    # The released CoWVLA VidTwin configuration produces two [8, 16, 7]
    # motion tensors. After concatenating x/y channels, each of the 16
    # temporal positions owns 7 * 16 = 112 values (16 * 112 = 1792).
    motion_latent_dim: int = 1792
    motion_camera_key: str | None = None

    # SmolW condition and z-flow branches.
    motion_projector_hidden_dim: int = 1024
    z_loss_weight: float = 1.0

    # Keep every episode frame as a training anchor. LeRobot clamps future
    # observations to the final frame; SmolW turns padded LIBERO actions into
    # stationary commands before normalization and supervises them as valid.
    drop_n_last_frames: int = 0
    # LIBERO actions use six relative pose deltas followed by an absolute
    # gripper command. A stationary tail zeros every dimension except these
    # hold dimensions, whose last in-episode value is retained. Negative
    # indices follow normal Python indexing.
    stationary_action_hold_dims: list[int] = field(default_factory=lambda: [-1])

    # TensorBoard logging is performed by lerobot-train on the main process.
    # Relative log directories are resolved below the training output dir.
    tensorboard_enable: bool = False
    tensorboard_log_dir: str | None = None
    tensorboard_log_freq: int = 100
    tensorboard_flush_secs: int = 30
    tensorboard_max_queue: int = 10
    tensorboard_filename_suffix: str = ""
    tensorboard_log_parameters: bool = False
    tensorboard_log_gradients: bool = False
    tensorboard_histogram_freq: int = 1_000

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.train_version not in {None, "A", "B", "C", "D"}:
            raise ValueError("train_version must be A, B, C, or D (None is reserved for legacy checkpoints).")
        if self.motion_horizon is None:
            self.motion_horizon = self.chunk_size
        if self.train_expert_only:
            raise ValueError("SmolW jointly trains the VLM condition path and action expert.")
        if self.motion_horizon <= 0:
            raise ValueError(f"motion_horizon must be positive, got {self.motion_horizon}.")
        if self.motion_horizon != self.chunk_size:
            raise ValueError(
                "SmolW currently requires motion_horizon == chunk_size so future motion and the action "
                f"chunk describe the same interval; got {self.motion_horizon} and {self.chunk_size}."
            )
        if self.memory_stride <= 0:
            raise ValueError(f"memory_stride must be positive, got {self.memory_stride}.")
        if self.vidtwin_num_frames != 16:
            raise ValueError(
                "The bundled VidTwin checkpoint architecture requires vidtwin_num_frames=16, "
                f"got {self.vidtwin_num_frames}."
            )
        if self.vidtwin_input_height <= 0 or self.vidtwin_input_width <= 0:
            raise ValueError(
                "vidtwin_input_height and vidtwin_input_width must be positive, got "
                f"{self.vidtwin_input_height}x{self.vidtwin_input_width}."
            )
        if self.motion_latent_dim <= 0:
            raise ValueError(f"motion_latent_dim must be positive, got {self.motion_latent_dim}.")
        if self.motion_latent_dim % self.vidtwin_num_frames != 0:
            raise ValueError(
                "motion_latent_dim must be divisible by vidtwin_num_frames for temporal z tokenization; got "
                f"{self.motion_latent_dim} and {self.vidtwin_num_frames}."
            )
        if self.motion_projector_hidden_dim <= 0:
            raise ValueError("SmolW motion projector hidden dimension must be positive.")
        if self.z_loss_weight <= 0:
            raise ValueError(
                f"Joint SmolW training requires z_loss_weight to be positive, got {self.z_loss_weight}."
            )
        if not self.use_cache:
            raise ValueError("SmolW requires use_cache=True for condition-prefix execution.")
        if self.vidtwin_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "vidtwin_dtype must be one of 'float32', 'float16', or 'bfloat16', got "
                f"{self.vidtwin_dtype!r}."
            )
        if self.tensorboard_log_freq <= 0:
            raise ValueError(f"tensorboard_log_freq must be positive, got {self.tensorboard_log_freq}.")
        if self.tensorboard_flush_secs <= 0:
            raise ValueError(f"tensorboard_flush_secs must be positive, got {self.tensorboard_flush_secs}.")
        if self.tensorboard_max_queue <= 0:
            raise ValueError(f"tensorboard_max_queue must be positive, got {self.tensorboard_max_queue}.")
        if self.tensorboard_histogram_freq <= 0:
            raise ValueError(
                f"tensorboard_histogram_freq must be positive, got {self.tensorboard_histogram_freq}."
            )

        if self.drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be non-negative, got {self.drop_n_last_frames}.")
        if any(not isinstance(index, int) for index in self.stationary_action_hold_dims):
            raise ValueError("stationary_action_hold_dims must contain only integer action indices.")
        if len(set(self.stationary_action_hold_dims)) != len(self.stationary_action_hold_dims):
            raise ValueError("stationary_action_hold_dims must not contain duplicate indices.")

    @property
    def past_motion_delta_indices(self) -> list[int]:
        """H observations ending at t, with the configured memory stride."""
        assert self.motion_horizon is not None
        return [
            -(self.motion_horizon - 1 - index) * self.memory_stride for index in range(self.motion_horizon)
        ]

    @property
    def future_motion_delta_indices(self) -> list[int]:
        """H future observations following t, aligned with H actions."""
        assert self.motion_horizon is not None
        return list(range(1, self.motion_horizon + 1))

    @property
    def motion_token_dim(self) -> int:
        """VidTwin values carried by one temporal position (1792 / 16 = 112)."""
        return self.motion_latent_dim // self.vidtwin_num_frames

    @property
    def uses_past_motion(self) -> bool:
        return self.train_version != "A"

    @property
    def uses_future_motion(self) -> bool:
        return self.train_version in {None, "C", "D"}

    @property
    def observation_delta_indices(self) -> list[int]:
        """Load only the observations required by the selected experiment."""
        if not self.uses_past_motion:
            return super().observation_delta_indices
        if not self.uses_future_motion:
            return self.past_motion_delta_indices
        return sorted(set(self.past_motion_delta_indices + self.future_motion_delta_indices))

    @property
    def current_observation_position(self) -> int:
        return self.observation_delta_indices.index(0)

    @property
    def past_motion_positions(self) -> list[int]:
        if not self.uses_past_motion:
            return []
        deltas = self.observation_delta_indices
        return [deltas.index(delta) for delta in self.past_motion_delta_indices]

    @property
    def future_motion_positions(self) -> list[int]:
        if not self.uses_future_motion:
            return []
        deltas = self.observation_delta_indices
        return [deltas.index(delta) for delta in self.future_motion_delta_indices]
