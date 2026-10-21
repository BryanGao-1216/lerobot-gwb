"""SmolVLA with a separate autoregressive latent-action pretraining stage."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .lam.specs import get_lam_spec


@PreTrainedConfig.register_subclass("smolvla_lap")
@dataclass
class SmolVLALAPConfig(SmolVLAConfig):
    training_stage: str = "post-training"
    lam_type: str = "univla"
    lam_model_path: str | None = None
    # A single short camera name (e.g. "front") or full dataset feature key.
    lam_camera_key: str | None = None
    lam_frame_gap: int = 12
    lam_batch_size: int = 8
    # None selects the official specification of lam_type. Resolved values are
    # saved in checkpoints so that vocabulary dimensions remain stable on reload.
    lam_num_tokens: int | None = None
    lam_codebook_size: int | None = None

    # Persist the vocabulary mapping alongside the policy, including across stages.
    latent_action_token_ids: list[int] = field(default_factory=list)
    latent_action_bos_token_id: int | None = None

    # Fresh LAP policies start from the original SmolVLM, never random VLM weights.
    load_vlm_weights: bool = True

    # Reuse lerobot-train's main-process TensorBoard logger for both stages.
    tensorboard_enable: bool = False
    tensorboard_log_dir: str | None = None
    tensorboard_log_freq: int = 10
    tensorboard_flush_secs: int = 30
    tensorboard_max_queue: int = 10
    tensorboard_filename_suffix: str = ""
    tensorboard_log_parameters: bool = False
    tensorboard_log_gradients: bool = False
    tensorboard_histogram_freq: int = 1_000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_stage not in {"pre-training", "post-training"}:
            raise ValueError("training_stage must be 'pre-training' or 'post-training'.")
        spec = get_lam_spec(self.lam_type)
        self.lam_type = spec.name
        if self.lam_frame_gap < 1 or self.lam_batch_size < 1:
            raise ValueError("lam_frame_gap and lam_batch_size must be positive.")
        if self.lam_num_tokens is None:
            self.lam_num_tokens = spec.num_tokens
        if self.lam_codebook_size is None:
            self.lam_codebook_size = spec.codebook_size
        if self.lam_num_tokens != spec.num_tokens or self.lam_codebook_size != spec.codebook_size:
            raise ValueError(
                f"{spec.name} uses {spec.num_tokens} tokens and a {spec.codebook_size}-entry codebook. "
                "Use a matching checkpoint/config or leave the token specification unset for a fresh run."
            )
        if self.latent_action_token_ids and len(self.latent_action_token_ids) != spec.codebook_size:
            raise ValueError("Checkpoint action vocabulary size does not match the selected LAM.")
        if self.n_obs_steps != 1:
            raise ValueError("SmolVLA LAP currently supports n_obs_steps=1.")
        for name in (
            "tensorboard_log_freq",
            "tensorboard_flush_secs",
            "tensorboard_max_queue",
            "tensorboard_histogram_freq",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if self.training_stage == "pre-training":
            if not self.lam_model_path:
                raise ValueError(
                    "pre-training requires policy.lam_model_path (checkpoint for the selected LAM)."
                )
            if not isinstance(self.lam_camera_key, str) or not self.lam_camera_key.strip():
                raise ValueError("pre-training requires exactly one policy.lam_camera_key.")
            if not self.load_vlm_weights and not self.pretrained_path:
                raise ValueError("Fresh pre-training must load the original pretrained SmolVLM weights.")
            if self.adapt_to_pi_aloha:
                raise ValueError(
                    "LAP pre-training uses the native dataset state; adapt_to_pi_aloha is unsupported."
                )

    def resolve_lam_camera_key(self, features: dict[str, Any]) -> str:
        """Resolve a short camera name without silently choosing another view."""
        name = self.lam_camera_key
        candidates = (name, f"observation.images.{name}")
        for key in candidates:
            if key in features:
                feature = features[key]
                if feature.get("dtype") not in {"image", "video"}:
                    continue
                shape = feature.get("shape") or ()
                if len(shape) != 3 or (feature.get("info") or {}).get("is_depth_map", False):
                    continue
                # Dataset metadata supports both HWC and CHW, even though the
                # reader returns CHW tensors in either case. Prefer named axes
                # so a height/width of 3 cannot disguise a non-RGB camera.
                names = feature.get("names") or ()
                if len(names) == 3 and names[2] in {"channel", "channels"}:
                    is_rgb = shape[2] == 3
                elif len(names) == 3 and names[0] in {"channel", "channels"}:
                    is_rgb = shape[0] == 3
                else:
                    is_rgb = shape[0] == 3 or shape[2] == 3
                if is_rgb:
                    return key
        available = {
            key: {"shape": value.get("shape"), "names": value.get("names")}
            for key, value in features.items()
            if value.get("dtype") in {"image", "video"}
        }
        raise ValueError(
            f"LAM camera {name!r} is not an RGB camera in this dataset. "
            f"Available image features (shape and axis names): {available}"
        )

    @property
    def action_delta_indices(self) -> list[int] | None:
        # There is no real-action supervision or need to decode action chunks here.
        if self.training_stage == "pre-training":
            return None
        return list(range(self.chunk_size))
