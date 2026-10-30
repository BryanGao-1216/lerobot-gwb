"""Strict CUDA FP32 UniVLA inference, kept outside the policy's state_dict and DDP."""

import zipfile
from pathlib import Path

import torch

ARCHITECTURE = {
    "in_dim": 3,
    "model_dim": 768,
    "latent_dim": 128,
    "num_latents": 16,
    "patch_size": 14,
    "enc_blocks": 12,
    "dec_blocks": 12,
    "num_heads": 12,
    "dropout": 0.0,
}


def load_lam_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"UniVLA LAM checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path))
    if not isinstance(state, dict):
        raise ValueError("Expected a UniVLA checkpoint dictionary.")
    for wrapper in ("state_dict", "model_state_dict"):
        if wrapper in state:
            state = state[wrapper]
            break
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(v, torch.Tensor) for v in state.values())
    ):
        raise ValueError("Expected a tensor state_dict or a state_dict/model_state_dict checkpoint wrapper.")
    state = {k.removeprefix("module.").removeprefix("lam."): v for k, v in state.items()}
    if "action_latent_controllable" not in state:
        raise ValueError("UniVLA requires the stage-2 task-centric LAM, not stage-1 or a VLA checkpoint.")
    codebook = state.get("vq_action.codebook.weight")
    if codebook is None or tuple(codebook.shape) != (16, 128):
        raise ValueError("UniVLA requires vq_action.codebook.weight with shape (16, 128).")
    return state


class UniVLALAMEncoder:
    """Plain object, so saving/moving/training the student never includes the teacher."""

    def __init__(self, checkpoint: str, device: torch.device):
        from .univla_model import ControllableDINOLatentActionModel

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("PI05 LAP pre-training with UniVLA requires CUDA.")
        state = load_lam_state(checkpoint)
        self.model = ControllableDINOLatentActionModel(**ARCHITECTURE)
        self.model.load_state_dict(state, strict=True, assign=True)
        del state
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[1:]) != (2, 3, 224, 224) or len(frames) == 0:
            raise ValueError(f"UniVLA expects frames [B,2,3,224,224], got {tuple(frames.shape)}.")
        frames = frames.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("UniVLA frames must contain finite RGB pixels in [0,1].")
        # Labels must be independent of the student's outer mixed-precision autocast.
        with torch.autocast(device_type="cuda", enabled=False):
            chunks = []
            for chunk in frames.split(8):
                tokens = self.model.vq_encode(chunk)["indices"].reshape(len(chunk), 4).long()
                if (tokens < 0).any() or (tokens >= 16).any():
                    raise ValueError("UniVLA must return four codebook indices in [0,15].")
                chunks.append(tokens)
        return torch.cat(chunks, dim=0)
