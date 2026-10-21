"""CUDA-only, frozen UniVLA stage-2 pseudo labels, independent of policy checkpoints."""

import torch

from .common import DINO_LAM_ARCHITECTURE, FrozenLAMEncoder


class UniVLALAMEncoder(FrozenLAMEncoder):
    lam_type = "univla"
    codebook_key = "vq_action.codebook.weight"

    def _validate_state(self, state: dict[str, torch.Tensor]) -> None:
        super()._validate_state(state)
        if "action_latent_controllable" not in state:
            raise ValueError("UniVLA requires the stage-2 task-centric LAM checkpoint, not stage-1 or a VLA.")

    def _build_model(self) -> torch.nn.Module:
        from .univla_model import ControllableDINOLatentActionModel

        return ControllableDINOLatentActionModel(**DINO_LAM_ARCHITECTURE)

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model.vq_encode(frames)["indices"].reshape(len(frames), self.spec.num_tokens).long()
