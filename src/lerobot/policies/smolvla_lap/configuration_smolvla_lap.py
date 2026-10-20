"""SmolVLA with a separate autoregressive latent-action pretraining stage."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


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
    lam_num_tokens: int = 4
    lam_codebook_size: int = 16

    # Persist the vocabulary mapping alongside the policy, including across stages.
    latent_action_token_ids: list[int] = field(default_factory=list)
    latent_action_bos_token_id: int | None = None

    # Fresh LAP policies start from the original SmolVLM, never random VLM weights.
    load_vlm_weights: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_stage not in {"pre-training", "post-training"}:
            raise ValueError("training_stage must be 'pre-training' or 'post-training'.")
        if self.lam_type != "univla":
            raise ValueError(f"Unsupported lam_type={self.lam_type!r}; currently only 'univla' is supported.")
        if self.lam_frame_gap < 1 or self.lam_batch_size < 1:
            raise ValueError("lam_frame_gap and lam_batch_size must be positive.")
        if self.lam_num_tokens != 4 or self.lam_codebook_size != 16:
            raise ValueError("The bundled UniVLA stage-2 model uses 4 tokens and a 16-entry codebook.")
        if self.n_obs_steps != 1:
            raise ValueError("SmolVLA LAP currently supports n_obs_steps=1.")
        if self.training_stage == "pre-training":
            if not self.lam_model_path:
                raise ValueError("pre-training requires policy.lam_model_path (UniVLA stage-2 checkpoint).")
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
                if feature.get("dtype") in {"image", "video"} and feature.get("shape", [0])[0] == 3:
                    return key
        raise ValueError(
            f"LAM camera {name!r} is not an RGB camera in this dataset. "
            f"Available image features: {[k for k, v in features.items() if v.get('dtype') in {'image', 'video'}]}"
        )

    @property
    def action_delta_indices(self) -> list[int] | None:
        # There is no real-action supervision or need to decode action chunks here.
        if self.training_stage == "pre-training":
            return None
        return list(range(self.chunk_size))
