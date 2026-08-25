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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("actionmem")
@dataclass
class ActionMemConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 10  # Must match the endpoint-effect tokenizer horizon
    n_action_steps: int = 10  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
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

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    # One of:
    # - "vlm_only": train action-token prediction without running the flow expert.
    # - "action_expert_only": train flow matching while freezing the PaliGemma VLM.
    # - "joint": train both objectives and both model branches.
    training_stage: str = "joint"
    # Deprecated compatibility alias for older PI0-style configs. When true it
    # selects training_stage="action_expert_only".
    train_expert_only: bool = False

    # Optimizer settings: see openpi `AdamW``
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

    tokenizer_max_length: int = 48  # see openpi `__post_init__`
    # Used only by the RLDS collator during training; inference uses the learned
    # classifier and starts flow matching from Gaussian noise.
    effect_tokenizer_checkpoint_path: str | None = None
    action_codebook_size: int = 256
    action_code_invalid_value: int = -1
    action_code_init_std: float = 0.02
    action_token_soft_target_temperature: float = 0.1
    action_condition_hidden_dim: int = 256
    action_condition_scale: float = 1.0
    flow_loss_weight: float = 1.0
    action_token_loss_weight: float = 0.1

    # TensorBoard settings. Logging is performed by lerobot-train on the main
    # process, while these policy fields keep ActionMem runs self-contained and
    # configurable through --policy.tensorboard_* CLI flags.
    tensorboard_enable: bool = False
    tensorboard_log_dir: str | None = None
    tensorboard_log_freq: int = 100
    tensorboard_flush_secs: int = 30
    tensorboard_max_queue: int = 10
    tensorboard_filename_suffix: str = ""
    tensorboard_log_parameters: bool = False
    tensorboard_log_gradients: bool = False
    tensorboard_histogram_freq: int = 1_000

    def __post_init__(self):
        super().__post_init__()

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
            raise ValueError(f"flow_loss_weight must be non-negative, got {self.flow_loss_weight}")
        if self.action_token_loss_weight < 0:
            raise ValueError(
                f"action_token_loss_weight must be non-negative, got {self.action_token_loss_weight}"
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

        if self.action_codebook_size < 2:
            raise ValueError(f"action_codebook_size must be at least 2, got {self.action_codebook_size}.")
        if self.action_code_init_std <= 0:
            raise ValueError(f"action_code_init_std must be positive, got {self.action_code_init_std}.")
        if self.action_token_soft_target_temperature <= 0:
            raise ValueError(
                "action_token_soft_target_temperature must be positive, got "
                f"{self.action_token_soft_target_temperature}."
            )
        if self.action_condition_hidden_dim <= 0:
            raise ValueError(
                f"action_condition_hidden_dim must be positive, got {self.action_condition_hidden_dim}."
            )
        if self.action_condition_scale < 0:
            raise ValueError(
                f"action_condition_scale must be non-negative, got {self.action_condition_scale}."
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

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
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
