"""Strict checkpoint loading and frozen FP32 pseudo-label inference."""

import zipfile
from pathlib import Path

import torch

from .specs import get_lam_spec

DINO_LAM_ARCHITECTURE = {
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
        raise FileNotFoundError(f"LAM checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path))
    if not isinstance(state, dict):
        raise ValueError("Expected a LAM checkpoint dictionary.")
    for wrapper in ("state_dict", "model_state_dict"):
        if wrapper in state:
            state = state[wrapper]
            break
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(v, torch.Tensor) for v in state.values())
    ):
        raise ValueError("Expected a tensor state_dict or a supported LAM checkpoint wrapper.")
    return {k.removeprefix("module.").removeprefix("lam."): v for k, v in state.items()}


class FrozenLAMEncoder:
    """Plain object: teacher weights stay outside the policy's DDP and state_dict."""

    lam_type: str
    codebook_key: str

    def __init__(self, checkpoint: str, device: torch.device, batch_size: int):
        self.spec = get_lam_spec(self.lam_type)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"{self.lam_type} LAP pre-training requires CUDA.")
        if batch_size < 1:
            raise ValueError("LAM inference batch_size must be positive.")
        state = load_lam_state(checkpoint)
        self._validate_state(state)
        self.model = self._build_model()
        # Bypass upstream overrides that silently set strict=False (notably LAPA).
        torch.nn.Module.load_state_dict(self.model, state, strict=True, assign=True)
        del state
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)
        self.batch_size = batch_size
        self._after_load()

    def _validate_state(self, state: dict[str, torch.Tensor]) -> None:
        codebook = state.get(self.codebook_key)
        shape = (self.spec.codebook_size, self.spec.latent_dim)
        if codebook is None or tuple(codebook.shape) != shape:
            raise ValueError(
                f"{self.lam_type} requires {self.codebook_key} with shape {shape}; "
                "check the selected LAM type and checkpoint."
            )

    def _build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def _after_load(self) -> None:
        pass

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        expected = (2, 3, self.spec.image_size, self.spec.image_size)
        if frames.ndim != 5 or tuple(frames.shape[1:]) != expected or frames.shape[0] < 1:
            raise ValueError(
                f"{self.lam_type} expects LAM frames [B,{','.join(map(str, expected))}], got {tuple(frames.shape)}."
            )
        frames = frames.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("LAM frames must have finite pixel values in [0,1].")
        # Outer VLM training autocast must never change a teacher's VQ assignments.
        with torch.autocast(device_type=self.device.type, enabled=False):
            chunks = []
            for chunk in frames.split(self.batch_size):
                tokens = self._encode(chunk)
                if tuple(tokens.shape) != (len(chunk), self.spec.num_tokens):
                    raise ValueError(
                        f"{self.lam_type} returned an invalid token shape: {tuple(tokens.shape)}."
                    )
                if (
                    tokens.dtype != torch.long
                    or (tokens < 0).any()
                    or (tokens >= self.spec.codebook_size).any()
                ):
                    raise ValueError(
                        f"{self.lam_type} must return integer code IDs in [0,{self.spec.codebook_size - 1}]."
                    )
                chunks.append(tokens)
        return torch.cat(chunks, dim=0)
