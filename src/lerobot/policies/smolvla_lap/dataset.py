"""Keep the teacher's raw frame pair separate from the student's observations."""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .lam.preprocessing import preprocess_lam_pair
from .lam.specs import get_lam_spec

LAM_IMAGE_PAIR = "observation.lap_image_pair"


class SmolVLALAPDataset(LeRobotDataset):
    def __init__(self, *args, policy_config, image_transforms=None, **kwargs):
        self._lap_pretraining = policy_config.training_stage == "pre-training"
        self._lap_image_size = get_lam_spec(policy_config.lam_type).image_size
        self._lap_image_transforms = image_transforms if self._lap_pretraining else None
        # Preserve raw teacher frames before applying any student augmentation.
        super().__init__(
            *args, image_transforms=None if self._lap_pretraining else image_transforms, **kwargs
        )
        self._lap_camera = (
            policy_config.resolve_lam_camera_key(self.meta.features) if self._lap_pretraining else None
        )

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        if not self._lap_pretraining:
            return item
        pair = item[self._lap_camera]
        # LeRobot's episode-aware reader already clamps t+H to the final frame.
        # Match each reference teacher's RGB PIL bilinear resize exactly, before
        # normalization or random augmentation, in the CPU data-loader workers.
        item[LAM_IMAGE_PAIR] = preprocess_lam_pair(pair, self._lap_image_size)
        item[self._lap_camera] = pair[:1]
        pad_key = f"{self._lap_camera}_is_pad"
        if pad_key in item:
            item[pad_key] = item[pad_key][:1]
        if self._lap_image_transforms is not None:
            for camera in self.meta.camera_keys:
                if camera not in self.meta.depth_keys:
                    item[camera] = self._lap_image_transforms(item[camera])
        return item
