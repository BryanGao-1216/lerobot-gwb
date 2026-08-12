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

"""Frozen q0-only decoder for the ActionMem action VQ-VAE."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _ceil_div_2(value: int) -> int:
    return (value + 1) // 2


def _choose_group_count(channels: int, requested_groups: int) -> int:
    for groups in range(min(channels, requested_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _squared_distance(x: Tensor, codebook: Tensor) -> Tensor:
    return (
        x.pow(2).sum(dim=-1, keepdim=True) - 2 * x @ codebook.t() + codebook.pow(2).sum(dim=-1).unsqueeze(0)
    )


def _pixel_shuffle_2d(x: Tensor, time_factor: int = 2, action_factor: int = 2) -> Tensor:
    if time_factor == 1 and action_factor == 1:
        return x
    batch, channels_times_factor, height, width = x.shape
    factor = time_factor * action_factor
    if channels_times_factor % factor != 0:
        raise ValueError(
            f"Channel count {channels_times_factor} is not divisible by "
            f"time_factor * action_factor = {factor}"
        )
    channels = channels_times_factor // factor
    x = x.view(batch, channels, time_factor, action_factor, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(batch, channels, height * time_factor, width * action_factor)


def _layer_count(layers_per_block: tuple[int, ...], index: int) -> int:
    if not layers_per_block:
        return 0
    return int(layers_per_block[min(index, len(layers_per_block) - 1)])


class CausalGroupNorm(nn.GroupNorm):
    """GroupNorm that does not mix statistics across the temporal axis."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            return super().forward(x)
        batch, channels, time, action = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * time, channels, action)
        x = super().forward(x)
        return x.view(batch, time, channels, action).permute(0, 2, 1, 3).contiguous()


class CausalConv2d(nn.Module):
    """2D convolution with causal time padding and symmetric action padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        pad_time = self.kernel_size[0] - 1
        pad_action_total = self.kernel_size[1] - 1
        pad_action_left = pad_action_total // 2
        pad_action_right = pad_action_total - pad_action_left
        x = F.pad(x, (pad_action_left, pad_action_right, pad_time, 0))
        return self.conv(x)


class ResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = CausalGroupNorm(
            _choose_group_count(in_channels, norm_groups),
            in_channels,
            eps=1e-6,
        )
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3)
        self.norm2 = CausalGroupNorm(
            _choose_group_count(out_channels, norm_groups),
            out_channels,
            eps=1e-6,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


class DownStage2D(nn.Module):
    """Encoder down block used by the trained action VQ-VAE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        downsample: bool,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(
                ResBlock2D(
                    current_channels,
                    out_channels,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
        self.blocks = nn.Sequential(*layers)
        self.downsample = (
            CausalConv2d(out_channels, out_channels, kernel_size=3, stride=2) if downsample else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(self.blocks(x))


class VQVLALikeEncoder(nn.Module):
    """Encoder architecture matching the action VQ-VAE training implementation."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        in_channels: int,
        latent_dim: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: tuple[int, ...],
        encoder_out_channels: int,
        norm_groups: int,
        dropout: float,
        num_res_blocks: int,
    ) -> None:
        super().__init__()
        if not block_out_channels:
            raise ValueError("block_out_channels must not be empty")

        self.horizon = horizon
        self.action_dim = action_dim
        self.in_channels = in_channels
        self.conv_in = CausalConv2d(in_channels, block_out_channels[0], kernel_size=3)

        stages: list[nn.Module] = []
        current_channels = block_out_channels[0]
        for index, out_channels in enumerate(block_out_channels):
            stages.append(
                DownStage2D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    num_layers=_layer_count(layers_per_block, index),
                    downsample=index < len(block_out_channels) - 1,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.mid_blocks = nn.Sequential(
            *[
                ResBlock2D(
                    block_out_channels[-1],
                    block_out_channels[-1],
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
                for _ in range(num_res_blocks)
            ]
        )
        self.norm_out = CausalGroupNorm(
            _choose_group_count(block_out_channels[-1], norm_groups),
            block_out_channels[-1],
            eps=1e-6,
        )
        self.conv_out = CausalConv2d(block_out_channels[-1], encoder_out_channels, kernel_size=3)

        bottleneck_height = horizon
        bottleneck_width = action_dim
        for _ in range(len(block_out_channels) - 1):
            bottleneck_height = _ceil_div_2(bottleneck_height)
            bottleneck_width = _ceil_div_2(bottleneck_width)
        flat_dim = encoder_out_channels * bottleneck_height * bottleneck_width
        self.to_latent = nn.Identity() if flat_dim == latent_dim else nn.Linear(flat_dim, latent_dim)

    def forward(self, action: Tensor) -> Tensor:
        if action.ndim != 4:
            raise ValueError(f"Expected encoded action input [B, C, T, A], got {tuple(action.shape)}")
        if action.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} encoder channels, got {action.shape[1]}")
        if action.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected action tail shape {(self.horizon, self.action_dim)}, got {tuple(action.shape[-2:])}"
            )
        x = self.conv_in(action)
        x = self.stages(x)
        x = self.mid_blocks(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return self.to_latent(x.reshape(x.shape[0], -1))


class OfficialLikeUpStage2D(nn.Module):
    """Decoder up block used by the trained action VQ-VAE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_layers + 1):
            layers.append(
                ResBlock2D(
                    current_channels,
                    out_channels,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
        self.blocks = nn.Sequential(*layers)
        self.upsample = CausalConv2d(out_channels, out_channels * 4, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.upsample(x)
        return _pixel_shuffle_2d(x, time_factor=2, action_factor=2)


class VQVLALikeDecoder(nn.Module):
    """Decoder architecture matching the action VQ-VAE training implementation."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        latent_dim: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: tuple[int, ...],
        encoder_out_channels: int,
        bottleneck_hw: tuple[int, int],
        norm_groups: int,
        dropout: float,
        num_res_blocks: int,
    ) -> None:
        super().__init__()
        if not block_out_channels:
            raise ValueError("block_out_channels must not be empty")

        self.horizon = horizon
        self.action_dim = action_dim
        self.encoder_out_channels = encoder_out_channels
        self.bottleneck_hw = bottleneck_hw
        self.from_latent = nn.Linear(
            latent_dim,
            encoder_out_channels * bottleneck_hw[0] * bottleneck_hw[1],
        )
        self.conv_in = CausalConv2d(
            encoder_out_channels,
            block_out_channels[-1],
            kernel_size=3,
        )

        self.mid_blocks = nn.Sequential(
            *[
                ResBlock2D(
                    block_out_channels[-1],
                    block_out_channels[-1],
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
                for _ in range(num_res_blocks)
            ]
        )

        reversed_channels = list(reversed(block_out_channels))
        reversed_layers = list(reversed(layers_per_block))
        stages: list[nn.Module] = []
        in_channels = reversed_channels[0]
        for index, out_channels in enumerate(reversed_channels[1:]):
            stages.append(
                OfficialLikeUpStage2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_layers=_layer_count(tuple(reversed_layers), index),
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.up_stages = nn.Sequential(*stages)

        upsample_factor = 2 ** max(len(block_out_channels) - 1, 0)
        decoder_width = bottleneck_hw[1] * upsample_factor
        self.projection = nn.Linear(decoder_width, action_dim)
        self.norm_out = CausalGroupNorm(
            _choose_group_count(in_channels, norm_groups),
            in_channels,
            eps=1e-6,
        )
        self.conv_out = CausalConv2d(in_channels, 1, kernel_size=3)

    def forward(self, latent: Tensor) -> Tensor:
        x = self.from_latent(latent)
        x = x.view(
            latent.shape[0],
            self.encoder_out_channels,
            self.bottleneck_hw[0],
            self.bottleneck_hw[1],
        )
        x = self.conv_in(x)
        x = self.mid_blocks(x)
        x = self.up_stages(x)
        if x.shape[-2] < self.horizon:
            x = F.pad(x, (0, 0, self.horizon - x.shape[-2], 0))
        x = self.projection(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x[:, 0, -self.horizon :, : self.action_dim]


class ActionVQVAEQ0Decoder(nn.Module):
    """Decode only the first residual codebook without adding q1-qN embeddings."""

    def __init__(
        self,
        decoder: VQVLALikeDecoder,
        q0_codebook: Tensor,
        action_mean: Tensor,
        action_std: Tensor,
        normalize_actions: bool,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.register_buffer("q0_codebook", q0_codebook)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.normalize_actions = normalize_actions
        self.horizon = decoder.horizon
        self.action_dim = decoder.action_dim
        self.codebook_size = q0_codebook.shape[0]

    def forward(self, q0_codes: Tensor) -> Tensor:
        q0_codes = q0_codes.to(device=self.q0_codebook.device, dtype=torch.long).reshape(-1)
        if torch.any((q0_codes < 0) | (q0_codes >= self.codebook_size)):
            invalid = q0_codes[(q0_codes < 0) | (q0_codes >= self.codebook_size)].detach().cpu().tolist()
            raise ValueError(f"q0 codes must be in [0, {self.codebook_size - 1}], got {invalid}.")

        # Residual VQ decoding is the sum of selected codebook embeddings.
        # ActionMem predicts only q0, so the coarse latent is exactly the q0
        # embedding; filling missing residual codes with index 0 would be wrong.
        latent = self.q0_codebook[q0_codes]
        actions = self.decoder(latent)
        if self.normalize_actions:
            actions = actions * self.action_std.view(1, 1, -1) + self.action_mean.view(1, 1, -1)
        return actions


class ActionVQVAEQ0Encoder(nn.Module):
    """Encode action chunks and return the nearest code from q0 only."""

    def __init__(
        self,
        encoder: VQVLALikeEncoder,
        q0_codebook: Tensor,
        time_emb: Tensor | None,
        xyz_emb: Tensor | None,
        euler_emb: Tensor | None,
        gripper_emb: Tensor | None,
        action_mean: Tensor,
        action_std: Tensor,
        normalize_actions: bool,
        use_action_type_pe: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("q0_codebook", q0_codebook)
        self.register_buffer("time_emb", time_emb)
        self.register_buffer("xyz_emb", xyz_emb)
        self.register_buffer("euler_emb", euler_emb)
        self.register_buffer("gripper_emb", gripper_emb)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.normalize_actions = normalize_actions
        self.use_action_type_pe = use_action_type_pe
        self.horizon = encoder.horizon
        self.action_dim = encoder.action_dim
        self.codebook_size = q0_codebook.shape[0]

    def _encode_latents(self, actions: Tensor) -> Tensor:
        actions = actions.to(device=self.q0_codebook.device, dtype=torch.float32)
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected action chunks [B, {self.horizon}, {self.action_dim}], got {tuple(actions.shape)}"
            )
        if self.normalize_actions:
            actions = (actions - self.action_mean.view(1, 1, -1)) / self.action_std.clamp_min(1e-6).view(
                1, 1, -1
            )

        encoder_input = actions.unsqueeze(1)
        if self.encoder.in_channels == 1:
            pass
        elif self.time_emb is None:
            raise ValueError("The VQ-VAE encoder expects positional channels, but time_emb is unavailable.")
        elif self.use_action_type_pe:
            if self.xyz_emb is None or self.euler_emb is None or self.gripper_emb is None:
                raise ValueError("Action-type positional embeddings are enabled but unavailable.")
            action_type_emb = torch.cat([self.xyz_emb, self.euler_emb, self.gripper_emb], dim=-1)
            encoder_input = encoder_input + action_type_emb.to(
                device=encoder_input.device, dtype=encoder_input.dtype
            )
        if self.encoder.in_channels != 1:
            time_emb = self.time_emb.permute(0, 2, 1, 3)
            encoder_input = encoder_input + time_emb.to(
                device=encoder_input.device, dtype=encoder_input.dtype
            )

        return self.encoder(encoder_input)

    def compute_code_distances(self, actions: Tensor) -> Tensor:
        """Return squared latent distances to every first-layer codebook center."""
        latents = self._encode_latents(actions)
        return _squared_distance(latents.float(), self.q0_codebook.float()).clamp_min_(0)

    def forward(self, actions: Tensor) -> Tensor:
        return self.compute_code_distances(actions).argmin(dim=-1)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a checkpoint mapping in {path}, got {type(checkpoint).__name__}.")
    return checkpoint


def load_action_vqvae_q0_decoder(checkpoint_path: str | Path) -> ActionVQVAEQ0Decoder:
    """Load only q0 and the frozen decoder from an action VQ-VAE checkpoint."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Action VQ-VAE checkpoint does not exist: {resolved_path}")

    checkpoint = _load_checkpoint(resolved_path)
    config = checkpoint.get("config")
    state_dict = checkpoint.get("model")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Action VQ-VAE checkpoint {resolved_path} must contain 'config' and 'model' mappings."
        )

    horizon = int(config["horizon"])
    action_dim = int(config["action_dim"])
    latent_dim = int(config.get("latent_dim", 128))
    block_out_channels = tuple(
        int(value)
        for value in config.get(
            "block_out_channels",
            (128, 256, 256, 512),
        )
    )
    layers_per_block = tuple(
        int(value)
        for value in config.get(
            "layers_per_block",
            (4, 4, 4, 4),
        )
    )
    encoder_out_channels = int(config.get("encoder_out_channels", 128))

    bottleneck_height = horizon
    bottleneck_width = action_dim
    for _ in range(len(block_out_channels) - 1):
        bottleneck_height = _ceil_div_2(bottleneck_height)
        bottleneck_width = _ceil_div_2(bottleneck_width)

    decoder = VQVLALikeDecoder(
        horizon=horizon,
        action_dim=action_dim,
        latent_dim=latent_dim,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        encoder_out_channels=encoder_out_channels,
        bottleneck_hw=(bottleneck_height, bottleneck_width),
        norm_groups=int(config.get("norm_groups", 32)),
        dropout=float(config.get("dropout", 0.0)),
        num_res_blocks=int(config.get("num_res_blocks", 4)),
    )

    decoder_state = {
        key.removeprefix("decoder."): value for key, value in state_dict.items() if key.startswith("decoder.")
    }
    if not decoder_state:
        raise ValueError(f"No decoder weights found in Action VQ-VAE checkpoint {resolved_path}.")
    try:
        decoder.load_state_dict(decoder_state, strict=True, assign=True)
    except TypeError:
        decoder.load_state_dict(decoder_state, strict=True)

    required_keys = (
        "quantizer.layers.0.codebook",
        "action_mean",
        "action_std",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        raise ValueError(f"Action VQ-VAE checkpoint {resolved_path} is missing keys: {missing_keys}.")

    q0_codebook = state_dict["quantizer.layers.0.codebook"].detach()
    expected_codebook_size = int(config.get("codebook_size", q0_codebook.shape[0]))
    if q0_codebook.shape != (expected_codebook_size, latent_dim):
        raise ValueError(
            f"Unexpected q0 codebook shape {tuple(q0_codebook.shape)} in {resolved_path}; "
            f"expected {(expected_codebook_size, latent_dim)}."
        )

    model = ActionVQVAEQ0Decoder(
        decoder=decoder,
        q0_codebook=q0_codebook,
        action_mean=state_dict["action_mean"].detach(),
        action_std=state_dict["action_std"].detach(),
        normalize_actions=bool(config.get("normalize_actions", False)),
    )
    model.requires_grad_(False)
    model.eval()
    return model


def load_action_vqvae_q0_encoder(checkpoint_path: str | Path) -> ActionVQVAEQ0Encoder:
    """Load the frozen encoder, positional embeddings, and first codebook."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Action VQ-VAE checkpoint does not exist: {resolved_path}")

    checkpoint = _load_checkpoint(resolved_path)
    config = checkpoint.get("config")
    state_dict = checkpoint.get("model")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Action VQ-VAE checkpoint {resolved_path} must contain 'config' and 'model' mappings."
        )

    horizon = int(config["horizon"])
    action_dim = int(config["action_dim"])
    latent_dim = int(config.get("latent_dim", 128))
    block_out_channels = tuple(int(value) for value in config.get("block_out_channels", (128, 256, 256, 512)))
    layers_per_block = tuple(int(value) for value in config.get("layers_per_block", (4, 4, 4, 4)))
    encoder_out_channels = int(config.get("encoder_out_channels", 128))
    encoder_conv_weight = state_dict.get("encoder.conv_in.conv.weight")
    if not isinstance(encoder_conv_weight, Tensor):
        raise ValueError(
            f"Action VQ-VAE checkpoint {resolved_path} is missing 'encoder.conv_in.conv.weight'."
        )
    encoder_in_channels = int(encoder_conv_weight.shape[1])
    time_emb = state_dict.get("time_emb")
    if encoder_in_channels != 1 and not isinstance(time_emb, Tensor):
        raise ValueError(f"Action VQ-VAE checkpoint {resolved_path} is missing 'time_emb'.")
    encoder = VQVLALikeEncoder(
        horizon=horizon,
        action_dim=action_dim,
        in_channels=encoder_in_channels,
        latent_dim=latent_dim,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        encoder_out_channels=encoder_out_channels,
        norm_groups=int(config.get("norm_groups", 32)),
        dropout=float(config.get("dropout", 0.0)),
        num_res_blocks=int(config.get("num_res_blocks", 4)),
    )
    encoder_state = {
        key.removeprefix("encoder."): value for key, value in state_dict.items() if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder weights found in Action VQ-VAE checkpoint {resolved_path}.")
    try:
        encoder.load_state_dict(encoder_state, strict=True, assign=True)
    except TypeError:
        encoder.load_state_dict(encoder_state, strict=True)

    required_keys = ("quantizer.layers.0.codebook", "action_mean", "action_std")
    if bool(config.get("use_action_type_pe", False)):
        required_keys += ("xyz_emb", "euler_emb", "gripper_emb")
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        raise ValueError(f"Action VQ-VAE checkpoint {resolved_path} is missing keys: {missing_keys}.")

    q0_codebook = state_dict["quantizer.layers.0.codebook"].detach()
    model = ActionVQVAEQ0Encoder(
        encoder=encoder,
        q0_codebook=q0_codebook,
        time_emb=time_emb.detach() if isinstance(time_emb, Tensor) else None,
        xyz_emb=state_dict["xyz_emb"].detach() if "xyz_emb" in state_dict else None,
        euler_emb=state_dict["euler_emb"].detach() if "euler_emb" in state_dict else None,
        gripper_emb=state_dict["gripper_emb"].detach() if "gripper_emb" in state_dict else None,
        action_mean=state_dict["action_mean"].detach(),
        action_std=state_dict["action_std"].detach(),
        normalize_actions=bool(config.get("normalize_actions", False)),
        use_action_type_pe=bool(config.get("use_action_type_pe", False)),
    )
    model.requires_grad_(False)
    model.eval()
    return model
