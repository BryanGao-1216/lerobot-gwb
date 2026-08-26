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

"""Configuration for Smol ActionMem with an independent action vocabulary."""

import logging
from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smol_actionmem")
@dataclass
class SmolActionMemConfig(SmolVLAConfig):
    """SmolVLA with a continuous flow condition derived from 256 action logits."""

    chunk_size: int = 10
    n_action_steps: int = 10
    # Future action indices past an episode boundary repeat its final frame, so
    # every episode frame remains a valid chunk start.
    drop_n_last_frames: int = 0
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    # Keep the tokenizer used for task strings separate from the VLM checkpoint.
    # The processor maps endpoint-effect codes to a model-local action embedding table;
    # these IDs never enter the SmolVLM tokenizer or language vocabulary.
    tokenizer_name: str | None = None
    # The endpoint-effect tokenizer is used only by the training data collator.
    # Inference consumes the learned classifier and does not load this file.
    effect_tokenizer_checkpoint_path: str | None = None
    action_codebook_size: int = 256
    action_code_invalid_value: int = -1
    # Initialization used only by the new action-code embedding and classifier.
    action_code_init_std: float = 0.02
    # Temperature of y_k = softmax(-||E(A) - e_k||^2 / T).
    # Spherical latent/codebook distances lie in [0, 4]; 0.1 provides an
    # informative soft target without reducing it to a hard one-hot label.
    action_token_soft_target_temperature: float = 0.1
    # Map normalized 256-way logits to residual FiLM parameters for every flow
    # prediction. The final projection is zero-initialized so an old checkpoint
    # starts from the unconditioned flow field and learns the new condition
    # smoothly.
    action_condition_hidden_dim: int = 256
    action_condition_scale: float = 1.0

    # ActionMem training objectives.
    flow_loss_weight: float = 1.0
    # Action-code KL is auxiliary: flow generation consumes predicted continuous logits,
    # never a ground-truth or argmax action token.
    action_token_loss_weight: float = 0.1
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

        # Older converted checkpoints stored horizon - 1 here and would keep
        # dropping tail chunks even after the dataset gained last-frame
        # padding. Normalize those configs when they are loaded.
        if self.drop_n_last_frames != 0:
            logging.warning(
                "Smol ActionMem ignores drop_n_last_frames=%d because tail action chunks are "
                "completed by repeating the episode's final frame.",
                self.drop_n_last_frames,
            )
            self.drop_n_last_frames = 0

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

        if self.action_code_init_std <= 0:
            raise ValueError(f"action_code_init_std must be positive, got {self.action_code_init_std}.")
        if self.action_codebook_size < 2:
            raise ValueError(
                f"action_codebook_size must be at least 2, got {self.action_codebook_size}."
            )
        if self.action_token_soft_target_temperature <= 0:
            raise ValueError(
                "action_token_soft_target_temperature must be positive, got "
                f"{self.action_token_soft_target_temperature}."
            )
        if self.action_condition_hidden_dim <= 0:
            raise ValueError(
                "action_condition_hidden_dim must be positive, got "
                f"{self.action_condition_hidden_dim}."
            )
        if self.action_condition_scale < 0:
            raise ValueError(
                f"action_condition_scale must be non-negative, got {self.action_condition_scale}."
            )

        for name in (
            "tensorboard_log_freq",
            "tensorboard_flush_secs",
            "tensorboard_max_queue",
            "tensorboard_histogram_freq",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
