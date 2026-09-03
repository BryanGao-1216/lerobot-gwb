#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen VidTwin motion extraction used by SmolW.

The checkpoint loader, preprocessing contract, and ``z_motion_x/y`` flattening
mirror ``scripts/CoWVLA``.  VidTwin stays outside the SmolW ``nn.Module`` tree,
so its frozen weights are neither optimized nor duplicated in policy
checkpoints.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_BUNDLED_VIDTWIN_CONFIG = (
    Path(__file__).parent / "vidtwin" / "configs" / "vidtwin_structure_7_7_8_dynamics_7_8.yaml"
)


def _get_obj_from_str(path: str):
    """Copy of CoWVLA's lightweight OmegaConf object resolver."""
    module, cls = path.rsplit(".", 1)
    importlib.invalidate_caches()
    return getattr(importlib.import_module(module, package=None), cls)


def _instantiate_from_config(config):
    """Instantiate a VidTwin object from the same ``target`` schema as CoWVLA."""
    if "target" not in config:
        raise KeyError("Expected a VidTwin config containing a 'target' entry.")
    return _get_obj_from_str(config["target"])(**config.get("params", {}))


def load_vidtwin_model_from_config(checkpoint_path: str | Path):
    """Load a checkpoint into SmolW's bundled VidTwin architecture."""
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError("SmolW VidTwin loading requires omegaconf.") from exc

    try:
        from safetensors.torch import load_file as load_safetensors
    except ImportError as exc:
        raise ImportError("SmolW VidTwin loading requires safetensors.") from exc

    config_path = _BUNDLED_VIDTWIN_CONFIG.resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"VidTwin config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VidTwin checkpoint not found: {checkpoint_path}")

    config = OmegaConf.load(config_path)
    model = _instantiate_from_config(config.model)

    suffix = checkpoint_path.suffix.lower()
    if suffix == ".ckpt":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
    elif suffix == ".safetensors":
        state_dict = load_safetensors(str(checkpoint_path))
    else:
        raise NotImplementedError(
            f"Unsupported VidTwin checkpoint extension {suffix!r}; expected .ckpt or .safetensors."
        )

    # CoWVLA ignores perceptual/adversarial loss weights when restoring the
    # inference autoencoder.
    state_dict = {key: value for key, value in state_dict.items() if not key.startswith("loss")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"VidTwin restored with {len(missing)} missing keys: {missing}")
    if unexpected:
        print(f"VidTwin restored with {len(unexpected)} unexpected keys: {unexpected}")
    return model


class VidTwinMotionExtractor:
    """Lazily load frozen VidTwin and extract flattened motion latents."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None,
        num_frames: int = 16,
        input_height: int = 224,
        input_width: int = 224,
        dtype: str = "bfloat16",
        sample_posterior: bool = False,
        expected_latent_dim: int = 1792,
    ) -> None:
        if checkpoint_path is None:
            raise ValueError(
                "SmolW requires vidtwin_checkpoint_path in the training/inference launch config."
            )
        if dtype not in _TORCH_DTYPES:
            raise ValueError(f"Unsupported VidTwin dtype: {dtype!r}.")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.num_frames = num_frames
        self.input_height = input_height
        self.input_width = input_width
        self.dtype = _TORCH_DTYPES[dtype]
        self.sample_posterior = sample_posterior
        self.expected_latent_dim = expected_latent_dim
        self._model = None
        self._device: torch.device | None = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("VidTwin is loaded lazily on the first encode call.")
        return self._model

    def _load(self, device: torch.device) -> None:
        model = load_vidtwin_model_from_config(self.checkpoint_path)
        model.eval()
        model.requires_grad_(False)
        # CoWVLA's YAML sets DiagonalGaussianRegularizer(sample=True).  Keep
        # that behavior configurable without changing the checkpoint.
        regularization = getattr(model, "regularization", None)
        if regularization is not None and hasattr(regularization, "sample"):
            regularization.sample = self.sample_posterior
        model.to(device=device, dtype=self.dtype)
        self._model = model
        self._device = device

    def _ensure_device(self, device: torch.device) -> None:
        if self._model is None:
            self._load(device)
        elif self._device != device:
            self.model.to(device=device)
            self._device = device

    def preprocess(self, frames: Tensor) -> Tensor:
        """Apply CoWVLA's 16-frame, center-crop, [-1, 1] preprocessing.

        Args:
            frames: ``[B, T, C, H, W]`` RGB frames in ``[0, 1]`` or uint8.

        Returns:
            VidTwin input with shape ``[B, C, num_frames, H, W]``.
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected motion frames [B,T,C,H,W], got {tuple(frames.shape)}.")
        if frames.shape[1] == 0:
            raise ValueError("Cannot extract VidTwin motion from an empty temporal window.")
        if frames.shape[2] != 3:
            raise ValueError(f"VidTwin expects RGB frames, got {frames.shape[2]} channels.")

        if frames.dtype == torch.uint8:
            frames = frames.to(dtype=torch.float32) / 255.0
        else:
            frames = frames.to(dtype=torch.float32)

        # CoWVLA uses torch.linspace(...).long() to select exactly 16 frames.
        indices = torch.linspace(
            0,
            frames.shape[1] - 1,
            self.num_frames,
            device=frames.device,
        ).long()
        frames = frames.index_select(1, indices)

        batch_size, frame_count, channels, height, width = frames.shape
        frames = frames.reshape(batch_size * frame_count, channels, height, width)

        # torchvision.transforms.Resize(input_height) preserves aspect ratio;
        # reproduce it with interpolate and then center crop.
        if height <= width:
            resized_height = self.input_height
            resized_width = max(self.input_width, round(width * self.input_height / height))
        else:
            resized_width = self.input_width
            resized_height = max(self.input_height, round(height * self.input_width / width))
        frames = F.interpolate(
            frames,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - self.input_height) // 2
        left = (resized_width - self.input_width) // 2
        frames = frames[
            :,
            :,
            top : top + self.input_height,
            left : left + self.input_width,
        ]
        frames = (frames - 0.5) / 0.5
        frames = frames.reshape(
            batch_size,
            frame_count,
            channels,
            self.input_height,
            self.input_width,
        )
        return frames.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def flatten_motion_latents(z_motion_x: Tensor, z_motion_y: Tensor) -> Tensor:
        """Copy CoWVLA's concat + ``b d f n -> b (f n d)`` flattening."""
        if z_motion_x.ndim != 4 or z_motion_y.ndim != 4:
            raise ValueError(
                "Expected VidTwin motion latents [B,D,F,N], got "
                f"{tuple(z_motion_x.shape)} and {tuple(z_motion_y.shape)}."
            )
        if z_motion_x.shape[0] != z_motion_y.shape[0] or z_motion_x.shape[2:] != z_motion_y.shape[2:]:
            raise ValueError(
                "VidTwin x/y motion latents must share batch, frame, and spatial dimensions; got "
                f"{tuple(z_motion_x.shape)} and {tuple(z_motion_y.shape)}."
            )
        motion = torch.cat([z_motion_x, z_motion_y], dim=1)
        # [B,D,F,N] -> [B,F,N,D] -> [B,F*N*D]
        return motion.permute(0, 2, 3, 1).reshape(motion.shape[0], -1)

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        video = self.preprocess(frames)
        self._ensure_device(video.device)
        video = video.to(dtype=self.dtype)

        autocast_enabled = video.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=video.device.type, dtype=self.dtype, enabled=autocast_enabled):
            _, _, z_motion_x, z_motion_y = self.model.encode(video)
        motion = self.flatten_motion_latents(z_motion_x, z_motion_y).detach().float()
        if motion.shape[-1] != self.expected_latent_dim:
            raise ValueError(
                "VidTwin motion latent dimension does not match SmolW config: "
                f"extracted {motion.shape[-1]}, expected {self.expected_latent_dim}."
            )
        return motion

    @torch.no_grad()
    def encode_pair(self, past_frames: Tensor, future_frames: Tensor) -> tuple[Tensor, Tensor]:
        """Encode past and future clips in one VidTwin batch."""
        if past_frames.shape[0] != future_frames.shape[0]:
            raise ValueError("Past and future motion clips must have the same batch size.")
        batch_size = past_frames.shape[0]
        motion = self.encode(torch.cat([past_frames, future_frames], dim=0))
        return motion[:batch_size], motion[batch_size:]
