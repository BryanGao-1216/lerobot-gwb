"""Sample an episode-clamped teacher pair without exposing future frames to PI05."""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .configuration_pi05_lap import LAM_IMAGE_PAIR
from .lam.preprocessing import preprocess_lam_pair


class PI05LAPDataset(LeRobotDataset):
    def __init__(self, *args, policy_config, image_transforms=None, **kwargs):
        self._student_transforms = image_transforms
        # Capture raw teacher pixels before any random student image transforms.
        super().__init__(*args, image_transforms=None, **kwargs)
        self._lam_camera = policy_config.resolve_lam_camera_key(self.meta.features)

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        pair = item[self._lam_camera]
        # The native reader clamps t+chunk_size to this episode's last frame.
        item[LAM_IMAGE_PAIR] = preprocess_lam_pair(pair, 224)
        item[self._lam_camera] = pair[0]
        pad_key = f"{self._lam_camera}_is_pad"
        if pad_key in item:
            item[pad_key] = item[pad_key][0]
        if self._student_transforms is not None:
            for camera in self.meta.camera_keys:
                if camera not in self.meta.depth_keys:
                    item[camera] = self._student_transforms(item[camera])
        return item
