from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def _policy_with_image_slots(*keys: str) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        image_features={key: object() for key in keys},
        resize_imgs_with_padding=None,
    )
    return policy


def test_prepare_images_preserves_missing_intermediate_camera_slot():
    policy = _policy_with_image_slots(
        "observation.images.image",
        "observation.images.image2",
        "observation.images.image3",
    )
    batch = {
        "observation.images.image": torch.full((2, 3, 4, 4), 0.75),
        "observation.images.image_padding_mask": torch.tensor([True, True]),
        "observation.images.image3": torch.full((2, 3, 4, 4), 0.25),
        "observation.images.image3_padding_mask": torch.tensor([True, False]),
    }

    images, masks = policy.prepare_images(batch)

    assert len(images) == 3
    assert torch.allclose(images[0], torch.full_like(images[0], 0.5))
    assert torch.equal(images[1], torch.full_like(images[1], -1.0))
    assert torch.allclose(images[2], torch.full_like(images[2], -0.5))
    assert torch.equal(masks[0], torch.tensor([True, True]))
    assert torch.equal(masks[1], torch.tensor([False, False]))
    assert torch.equal(masks[2], torch.tensor([True, False]))


def test_prepare_images_resolves_temporal_image_and_mask_to_latest_step():
    policy = _policy_with_image_slots("observation.images.image")
    batch = {
        "observation.images.image": torch.tensor([0.25, 0.75]).reshape(1, 2, 1, 1, 1),
        "observation.images.image_padding_mask": torch.tensor([[False, True]]),
    }

    images, masks = policy.prepare_images(batch)

    assert torch.equal(images[0], torch.tensor([[[[0.5]]]]))
    assert torch.equal(masks[0], torch.tensor([True]))
