"""CUDA-only, frozen UniVLA stage-2 pseudo labels, independent of policy checkpoints."""

import zipfile
from pathlib import Path

import torch


class UniVLALAMEncoder:
    """A plain object: the teacher is not registered in the policy's DDP/state_dict."""

    def __init__(self, checkpoint: str, device: torch.device, batch_size: int):
        from .univla_model import ControllableDINOLatentActionModel

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("UniVLA LAP pre-training requires CUDA.")
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"UniVLA stage-2 checkpoint not found: {checkpoint}")
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(checkpoint)
        )
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
            raise ValueError("Expected a UniVLA tensor state_dict or a supported checkpoint wrapper.")
        state = {k.removeprefix("module.").removeprefix("lam."): v for k, v in state.items()}
        if "vq_action.codebook.weight" not in state or "action_latent_controllable" not in state:
            raise ValueError("UniVLA requires the stage-2 task-centric LAM checkpoint, not stage-1 or a VLA.")
        if state["vq_action.codebook.weight"].shape != (16, 128):
            raise ValueError("Expected UniVLA's 16 x 128 task-centric action codebook.")
        self.model = ControllableDINOLatentActionModel(
            in_dim=3,
            model_dim=768,
            latent_dim=128,
            num_latents=16,
            patch_size=14,
            enc_blocks=12,
            dec_blocks=12,
            num_heads=12,
            dropout=0.0,
        )
        # Full strict loading includes DINOv2, both branches and the decoder.
        self.model.load_state_dict(state, strict=True, assign=True)
        del state
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:] != (2, 3, 224, 224):
            raise ValueError(f"Expected LAM frames [B,2,3,224,224], got {tuple(frames.shape)}.")
        frames = frames.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("LAM frames must have finite pixel values in [0,1].")
        # Disable the outer training autocast: FP16/BF16 may change discrete VQ assignments.
        with torch.autocast(device_type="cuda", enabled=False):
            tokens = [
                self.model.vq_encode(chunk)["indices"].reshape(len(chunk), 4).long()
                for chunk in frames.split(self.batch_size)
            ]
        return torch.cat(tokens, dim=0)
