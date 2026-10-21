"""Reference RGB frame-pair preprocessing, before student-side augmentation."""

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def preprocess_lam_pair(pair: torch.Tensor, image_size: int) -> torch.Tensor:
    if pair.ndim != 4 or tuple(pair.shape[:2]) != (2, 3):
        raise ValueError(f"Expected two RGB frames for LAM, got {tuple(pair.shape)}.")
    return (
        torch.stack(
            [
                pil_to_tensor(to_pil_image(frame).resize((image_size, image_size), Image.Resampling.BILINEAR))
                for frame in pair
            ]
        )
        .float()
        .div_(255.0)
    )
