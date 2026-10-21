"""Frozen MVP-LAM teacher, using the bundled DINOv2 and original encoder/VQ."""

import torch

from .common import DINO_LAM_ARCHITECTURE, FrozenLAMEncoder


class MVPLAMEncoder(FrozenLAMEncoder):
    lam_type = "mvp_lam"
    codebook_key = "vq.codebook.weight"

    def _build_model(self) -> torch.nn.Module:
        from .mvp_model import DINOLatentActionModel

        return DINOLatentActionModel(**DINO_LAM_ARCHITECTURE)

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model.vq_encode(frames)["indices"].reshape(len(frames), self.spec.num_tokens).long()
