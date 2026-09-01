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

"""Configuration for SmolW, a VidTwin-motion-conditioned SmolVLA policy."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolw")
@dataclass
class SmolWConfig(SmolVLAConfig):
    """SmolVLA with a past-motion query and a predicted future-motion condition.

    The VLM still receives only the current observation.  Temporal frames are
    requested from a LeRobot dataset through ``observation_delta_indices`` and
    are consumed exclusively by the frozen VidTwin motion extractor.
    """

    # Motion/action temporal contract. ``None`` keeps the motion horizon tied to
    # the SmolVLA action chunk size.
    motion_horizon: int | None = None
    memory_stride: int = 1

    # The frozen VidTwin extractor is external to the policy state dict.  All
    # paths are supplied by the launch script or a saved SmolW config.
    vidtwin_repo_path: str | None = None
    vidtwin_config_path: str | None = None
    vidtwin_checkpoint_path: str | None = None
    vidtwin_num_frames: int = 16
    vidtwin_input_height: int = 224
    vidtwin_input_width: int = 224
    vidtwin_dtype: str = "bfloat16"
    vidtwin_sample_posterior: bool = True

    # The released CoWVLA VidTwin configuration produces two [8, 16, 7]
    # motion tensors, which are concatenated and flattened to 1792 values.
    motion_latent_dim: int = 1792
    motion_camera_key: str | None = None

    # New SmolW branches.
    motion_projector_hidden_dim: int = 1024
    motion_condition_hidden_dim: int = 1024
    motion_condition_scale: float = 1.0
    motion_loss_weight: float = 0.1
    detach_motion_condition: bool = False

    # A full future video is required for VidTwin supervision.  ``None`` drops
    # exactly the final H-1 episode frames in the LeRobot sampler.
    drop_n_last_frames: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.motion_horizon is None:
            self.motion_horizon = self.chunk_size
        if self.motion_horizon <= 0:
            raise ValueError(f"motion_horizon must be positive, got {self.motion_horizon}.")
        if self.motion_horizon != self.chunk_size:
            raise ValueError(
                "SmolW currently requires motion_horizon == chunk_size so future motion and the action "
                f"chunk describe the same interval; got {self.motion_horizon} and {self.chunk_size}."
            )
        if self.memory_stride <= 0:
            raise ValueError(f"memory_stride must be positive, got {self.memory_stride}.")
        if self.vidtwin_num_frames <= 0:
            raise ValueError(f"vidtwin_num_frames must be positive, got {self.vidtwin_num_frames}.")
        if self.vidtwin_input_height <= 0 or self.vidtwin_input_width <= 0:
            raise ValueError(
                "vidtwin_input_height and vidtwin_input_width must be positive, got "
                f"{self.vidtwin_input_height}x{self.vidtwin_input_width}."
            )
        if self.motion_latent_dim <= 0:
            raise ValueError(f"motion_latent_dim must be positive, got {self.motion_latent_dim}.")
        if self.motion_projector_hidden_dim <= 0 or self.motion_condition_hidden_dim <= 0:
            raise ValueError("SmolW motion projector hidden dimensions must be positive.")
        if self.motion_condition_scale < 0:
            raise ValueError(
                f"motion_condition_scale must be non-negative, got {self.motion_condition_scale}."
            )
        if self.motion_loss_weight < 0:
            raise ValueError(f"motion_loss_weight must be non-negative, got {self.motion_loss_weight}.")
        if not self.use_cache:
            raise ValueError("SmolW requires use_cache=True for motion-first prefix execution.")
        if self.vidtwin_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "vidtwin_dtype must be one of 'float32', 'float16', or 'bfloat16', got "
                f"{self.vidtwin_dtype!r}."
            )

        required_tail_drop = self.motion_horizon - 1
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = required_tail_drop
        elif self.drop_n_last_frames < required_tail_drop:
            raise ValueError(
                "SmolW future-motion supervision requires dropping at least motion_horizon - 1 "
                f"episode-tail frames; got drop_n_last_frames={self.drop_n_last_frames}, "
                f"required>={required_tail_drop}."
            )

    @property
    def past_motion_delta_indices(self) -> list[int]:
        """H observations ending at t, with the configured memory stride."""
        assert self.motion_horizon is not None
        return [
            -(self.motion_horizon - 1 - index) * self.memory_stride for index in range(self.motion_horizon)
        ]

    @property
    def future_motion_delta_indices(self) -> list[int]:
        """H contiguous observations starting at t."""
        assert self.motion_horizon is not None
        return list(range(self.motion_horizon))

    @property
    def observation_delta_indices(self) -> list[int]:
        """Union of history and future frames requested from LeRobot datasets."""
        return sorted(set(self.past_motion_delta_indices + self.future_motion_delta_indices))

    @property
    def current_observation_position(self) -> int:
        return self.observation_delta_indices.index(0)

    @property
    def past_motion_positions(self) -> list[int]:
        deltas = self.observation_delta_indices
        return [deltas.index(delta) for delta in self.past_motion_delta_indices]

    @property
    def future_motion_positions(self) -> list[int]:
        deltas = self.observation_delta_indices
        return [deltas.index(delta) for delta in self.future_motion_delta_indices]
