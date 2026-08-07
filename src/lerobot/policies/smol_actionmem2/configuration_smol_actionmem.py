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

"""Configuration for Smol ActionMem 2 with an independent action vocabulary."""

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smol_actionmem2")
@dataclass
class SmolActionMem2Config(SmolVLAConfig):
    """SmolVLA with a 256-way action classifier and independent action embeddings."""

    chunk_size: int = 16
    n_action_steps: int = 16
    drop_n_last_frames: int = 15
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    # Keep the tokenizer used for task strings separate from the VLM checkpoint.
    # The processor maps q0 codes to IDs in a model-local action embedding table;
    # these IDs never enter the SmolVLM tokenizer or language vocabulary.
    tokenizer_name: str | None = None
    action_token_map_path: str | None = None
    action_vqvae_checkpoint_path: str | None = None
    # Initialization used only by the new action-code embedding and classifier.
    action_code_init_std: float = 0.02

    # The VQ-VAE reconstruction is in VQ-VLA's BOUNDS_Q99 space. Restore the
    # OXE action first (while preserving masked gripper dimensions), then map
    # it into the flow target's MEAN_STD space.
    action_vqvae_input_q01: list[float] | None = None
    action_vqvae_input_q99: list[float] | None = None
    action_vqvae_input_mask: list[bool] | None = None
    action_vqvae_flow_mean: list[float] | None = None
    action_vqvae_flow_std: list[float] | None = None
    action_vqvae_flow_normalization_eps: float = 1e-8

    # ActionMem training objectives.
    flow_loss_weight: float = 1.0
    action_token_loss_weight: float = 1.0
    training_stage: str = "joint"

    # Compatibility with the original SmolVLA and PI0-style configurations.
    # When true it selects action_expert_only.
    train_expert_only: bool = False
    gradient_checkpointing: bool = False

    # Relative-action processing is kept aligned with ActionMem.
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None

    # TensorBoard is implemented generically in lerobot-train and reads these
    # policy fields.
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

        valid_training_stages = {"vlm_only", "action_expert_only", "joint"}
        if self.train_expert_only:
            if self.training_stage not in {"joint", "action_expert_only"}:
                raise ValueError(
                    f"train_expert_only=True conflicts with training_stage={self.training_stage!r}."
                )
            self.training_stage = "action_expert_only"
        if self.training_stage not in valid_training_stages:
            raise ValueError(
                f"training_stage must be one of {sorted(valid_training_stages)}, got {self.training_stage!r}."
            )

        if self.flow_loss_weight < 0:
            raise ValueError(f"flow_loss_weight must be non-negative, got {self.flow_loss_weight}.")
        if self.action_token_loss_weight < 0:
            raise ValueError(
                f"action_token_loss_weight must be non-negative, got {self.action_token_loss_weight}."
            )
        if self.training_stage == "vlm_only" and self.action_token_loss_weight == 0:
            raise ValueError("vlm_only training requires action_token_loss_weight > 0.")
        if self.training_stage == "action_expert_only" and self.flow_loss_weight == 0:
            raise ValueError("action_expert_only training requires flow_loss_weight > 0.")
        if (
            self.training_stage == "joint"
            and self.flow_loss_weight == 0
            and self.action_token_loss_weight == 0
        ):
            raise ValueError("joint training requires at least one non-zero loss weight.")

        if (self.action_vqvae_input_q01 is None) != (self.action_vqvae_input_q99 is None):
            raise ValueError(
                "action_vqvae_input_q01 and action_vqvae_input_q99 must either both be set or both be None."
            )
        if self.action_vqvae_input_q01 is not None:
            if len(self.action_vqvae_input_q01) != len(self.action_vqvae_input_q99):
                raise ValueError("action_vqvae_input_q01 and action_vqvae_input_q99 must match in length.")
            if self.action_vqvae_input_mask is None or len(self.action_vqvae_input_mask) != len(
                self.action_vqvae_input_q01
            ):
                raise ValueError(
                    "action_vqvae_input_mask must be set and match the q01/q99 dimension."
                )
        if (self.action_vqvae_flow_mean is None) != (self.action_vqvae_flow_std is None):
            raise ValueError(
                "action_vqvae_flow_mean and action_vqvae_flow_std must either both be set or both be None."
            )
        if self.action_vqvae_flow_mean is not None and len(self.action_vqvae_flow_mean) != len(
            self.action_vqvae_flow_std
        ):
            raise ValueError("action_vqvae_flow_mean and action_vqvae_flow_std must have the same length.")
        if self.action_vqvae_flow_normalization_eps <= 0:
            raise ValueError(
                "action_vqvae_flow_normalization_eps must be positive, got "
                f"{self.action_vqvae_flow_normalization_eps}."
            )
        if self.action_code_init_std <= 0:
            raise ValueError(f"action_code_init_std must be positive, got {self.action_code_init_std}.")

        for name in (
            "tensorboard_log_freq",
            "tensorboard_flush_secs",
            "tensorboard_max_queue",
            "tensorboard_histogram_freq",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
