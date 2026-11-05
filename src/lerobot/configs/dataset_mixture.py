"""Configuration for named, heterogeneous LeRobot training sources."""

import math
from dataclasses import dataclass, field


@dataclass
class VectorFieldConfig:
    key: str
    # None keeps every component; otherwise select/reorder before concatenation.
    indices: list[int] | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Vector field key must not be empty.")
        if self.indices is not None and (
            not self.indices
            or any(i < 0 for i in self.indices)
            or len(set(self.indices)) != len(self.indices)
        ):
            raise ValueError(f"{self.key}: indices must be a nonempty list of distinct nonnegative integers.")


@dataclass
class DatasetSourceConfig:
    repo_id: str
    root: str | None = None
    weight: float = 1.0
    # Convert stored absolute actions to offsets before normalization.
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Policy-owned OpenPI image augmentation, applied only during training.
    image_augmentation: bool = True
    # Evaluation uses pose-only errors for discrete grippers, all action dimensions for continuous ones.
    gripper_representation: str = "discrete"  # "discrete" or "continuous"
    action_fields: list[VectorFieldConfig] = field(default_factory=lambda: [VectorFieldConfig("action")])
    state_fields: list[VectorFieldConfig] = field(
        default_factory=lambda: [VectorFieldConfig("observation.state")]
    )
    # Model view -> source camera. None/empty values reserve masked, unloaded views.
    camera_fields: dict[str, str | None] = field(default_factory=dict)
    # One selected source camera for latent supervision; None uses the first nonempty view.
    lam_camera_key: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    video_backend: str | None = None
    tolerance_s: float | None = None
    eval_split: float | None = None

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("Each dataset needs a nonempty repo_id.")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(f"{self.repo_id}: weight must be finite and positive.")
        if not isinstance(self.use_relative_actions, bool) or not isinstance(self.image_augmentation, bool):
            raise ValueError("use_relative_actions and image_augmentation must be booleans.")
        if self.gripper_representation not in ("discrete", "continuous"):
            raise ValueError("gripper_representation must be 'discrete' or 'continuous'.")
        if not self.action_fields or not self.state_fields:
            raise ValueError("Each dataset needs action_fields and state_fields.")
        if any(
            not k.startswith("observation.images.")
            or not k.removeprefix("observation.images.").strip()
            or (v is not None and not isinstance(v, str))
            for k, v in self.camera_fields.items()
        ):
            raise ValueError(
                "camera_fields must map observation.images.<name> to source camera keys or null."
            )
        self.camera_fields = {
            k: (v.strip() or None) if v is not None else None for k, v in self.camera_fields.items()
        }
        selected_cameras = [key for key in self.camera_fields.values() if key is not None]
        if len(set(selected_cameras)) != len(selected_cameras):
            raise ValueError("A source camera may only map to one model camera.")
        if self.lam_camera_key is not None and (
            not isinstance(self.lam_camera_key, str)
            or not self.lam_camera_key.strip()
            or any(char in self.lam_camera_key for char in ",[]")
        ):
            raise ValueError("lam_camera_key must name exactly one camera.")
        if self.lam_camera_key is not None:
            self.resolve_lam_camera_key()
        if self.episodes is not None and (
            not self.episodes
            or any(i < 0 for i in self.episodes)
            or len(set(self.episodes)) != len(self.episodes)
        ):
            raise ValueError("episodes must be a nonempty list of distinct nonnegative integers.")
        if self.eval_split is not None and not 0 <= self.eval_split < 1:
            raise ValueError("eval_split must be in [0, 1).")
        if self.tolerance_s is not None and (not math.isfinite(self.tolerance_s) or self.tolerance_s <= 0):
            raise ValueError("tolerance_s must be finite and positive.")

    def resolve_lam_camera_key(self) -> str:
        """Select exactly one of the student's nonempty source cameras."""
        selected = [key for key in self.camera_fields.values() if key is not None]
        if not selected:
            raise ValueError(f"{self.repo_id}: latent pre-training requires a nonempty camera_fields view.")
        if self.lam_camera_key is None:
            return selected[0]
        key = self.lam_camera_key.strip()
        if key in selected:
            return key
        # Accept older configs that used a model-view alias, but never an unselected camera.
        source_key = self.camera_fields.get(key)
        if source_key is not None and source_key in selected:
            return source_key
        raise ValueError(
            f"{self.repo_id}: lam_camera_key {key!r} must select a nonempty VLM view from camera_fields. "
            f"Selected source cameras: {selected}."
        )
