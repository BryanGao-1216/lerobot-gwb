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

"""Frozen encoder for the endpoint-effect tokenizer shared by ActionMem policies.

The checkpoint contract is defined by ``scripts/effectTokenizer``.  That
tokenizer is trained on per-dataset q01/q99-normalized OXE action chunks, but
its MLP does not consume a flattened chunk.  It consumes a seven-dimensional
endpoint-effect descriptor: XYZ/RPY are accumulated over the horizon and the
gripper component is final-minus-initial.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

EFFECT_TOKENIZER_ARTIFACT_VERSION = 3
EFFECT_TOKENIZER_MODEL_TYPE = "mlp_effect_vqvae"
EFFECT_DESCRIPTOR_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "delta_gripper",
)
EFFECT_NORMALIZATION_CONTRACT = "per_dataset_q01_q99_to_minus1_plus1_except_gripper"


@dataclass(frozen=True)
class EffectTokenizerMetadata:
    checkpoint_path: str
    horizon: int
    action_dim: int
    target_control_hz: float | None
    codebook_size: int


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.GELU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def compute_effect_descriptors(actions: Tensor) -> Tensor:
    """Apply the exact endpoint-effect contract to ``[B, horizon, 7]`` chunks."""
    if actions.ndim != 3 or actions.shape[-1] != len(EFFECT_DESCRIPTOR_NAMES):
        raise ValueError(
            "Effect tokenizer expects action chunks shaped "
            f"[B, horizon, {len(EFFECT_DESCRIPTOR_NAMES)}], got {tuple(actions.shape)}."
        )
    if actions.shape[1] <= 0:
        raise ValueError("Effect tokenizer action chunks must contain at least one timestep.")

    # The reference NumPy implementation accumulates motion in float64 before
    # returning float32.  Keeping that detail avoids label changes for samples
    # very close to a Voronoi boundary.
    motion = actions[..., :6].to(torch.float64).sum(dim=-2).to(torch.float32)
    gripper = (actions[..., -1, 6] - actions[..., 0, 6]).to(torch.float32).unsqueeze(-1)
    return torch.cat((motion, gripper), dim=-1)


def _load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid effect-tokenizer checkpoint at {path}.")
    return checkpoint


def _validate_checkpoint_contract(checkpoint: Mapping[str, Any], path: Path) -> None:
    artifact_version = int(checkpoint.get("artifact_version", 0))
    if artifact_version != EFFECT_TOKENIZER_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported effect-tokenizer artifact version {artifact_version} in {path}; "
            f"expected {EFFECT_TOKENIZER_ARTIFACT_VERSION}."
        )
    if checkpoint.get("model_type") != EFFECT_TOKENIZER_MODEL_TYPE:
        raise ValueError(
            f"Checkpoint {path} has model_type={checkpoint.get('model_type')!r}; "
            f"expected {EFFECT_TOKENIZER_MODEL_TYPE!r}."
        )
    descriptor_names = tuple(checkpoint.get("descriptor_names", ()))
    if descriptor_names != EFFECT_DESCRIPTOR_NAMES:
        raise ValueError(
            f"Effect-tokenizer descriptor contract in {path} is {descriptor_names}, "
            f"expected {EFFECT_DESCRIPTOR_NAMES}."
        )


def load_effect_tokenizer_metadata(checkpoint_path: str | Path) -> EffectTokenizerMetadata:
    """Read the data/codebook contract without constructing the MLP."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Effect-tokenizer checkpoint does not exist: {resolved_path}")
    checkpoint = _load_trusted_checkpoint(resolved_path)
    _validate_checkpoint_contract(checkpoint, resolved_path)
    model_config = checkpoint.get("model_config")
    saved_config = checkpoint.get("config")
    if not isinstance(model_config, Mapping):
        raise ValueError(f"Effect-tokenizer checkpoint {resolved_path} is missing model_config.")
    if not isinstance(saved_config, Mapping) or not isinstance(saved_config.get("data"), Mapping):
        raise ValueError(f"Effect-tokenizer checkpoint {resolved_path} is missing config.data.")
    data_config = saved_config["data"]
    normalization = data_config.get("action_normalization")
    if normalization != EFFECT_NORMALIZATION_CONTRACT:
        raise ValueError(
            f"Effect-tokenizer checkpoint {resolved_path} uses action_normalization={normalization!r}; "
            f"expected {EFFECT_NORMALIZATION_CONTRACT!r}."
        )
    target_control_hz_raw = data_config.get("target_control_hz")
    return EffectTokenizerMetadata(
        checkpoint_path=str(resolved_path),
        horizon=int(data_config["horizon"]),
        action_dim=int(data_config["action_dim"]),
        target_control_hz=None if target_control_hz_raw is None else float(target_control_hz_raw),
        codebook_size=int(model_config.get("codebook_size", 256)),
    )


class EffectVQVAEActionEncoder(nn.Module):
    """Encode normalized action chunks and return all spherical-code distances."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        codebook: Tensor,
        horizon: int,
        action_dim: int,
        target_control_hz: float | None,
        gripper_weight: float,
        effect_scale: Tensor,
        normalize_latents: bool,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"Effect-tokenizer horizon must be positive, got {horizon}.")
        if action_dim != len(EFFECT_DESCRIPTOR_NAMES):
            raise ValueError(
                f"Effect tokenizer requires action_dim={len(EFFECT_DESCRIPTOR_NAMES)}, got {action_dim}."
            )
        if codebook.ndim != 2 or codebook.shape[0] < 2:
            raise ValueError(f"Invalid effect-tokenizer codebook shape {tuple(codebook.shape)}.")
        if gripper_weight <= 0:
            raise ValueError(f"Effect-tokenizer gripper_weight must be positive, got {gripper_weight}.")
        if effect_scale.shape != (len(EFFECT_DESCRIPTOR_NAMES),) or torch.any(effect_scale <= 0):
            raise ValueError(
                f"effect_scale must contain {len(EFFECT_DESCRIPTOR_NAMES)} positive values."
            )

        self.encoder = encoder
        self.register_buffer("codebook", codebook.detach().to(torch.float32))
        self.register_buffer("effect_scale", effect_scale.detach().to(torch.float32))
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.target_control_hz = target_control_hz
        self.gripper_weight = float(gripper_weight)
        self.normalize_latents = bool(normalize_latents)
        self.codebook_size = int(codebook.shape[0])

    def _encode_latents(self, actions: Tensor) -> Tensor:
        actions = actions.to(device=self.codebook.device, dtype=torch.float32)
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected action chunks [B, {self.horizon}, {self.action_dim}], got {tuple(actions.shape)}."
            )
        effects = compute_effect_descriptors(actions)
        effects = effects * self.effect_scale
        effects[..., -1] *= self.gripper_weight
        latents = self.encoder(effects)
        if self.normalize_latents:
            latents = F.normalize(latents.float(), dim=-1, eps=1e-6)
        return latents.float()

    def normalized_codebook(self) -> Tensor:
        if self.normalize_latents:
            return F.normalize(self.codebook.float(), dim=-1, eps=1e-6)
        return self.codebook.float()

    def compute_code_distances(self, actions: Tensor) -> Tensor:
        latents = self._encode_latents(actions)
        centers = self.normalized_codebook()
        return (
            latents.square().sum(dim=-1, keepdim=True)
            + centers.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * latents @ centers.t()
        ).clamp_min_(0.0)

    def forward(self, actions: Tensor) -> Tensor:
        return self.compute_code_distances(actions).argmin(dim=-1)


def load_effect_vqvae_action_encoder(checkpoint_path: str | Path) -> EffectVQVAEActionEncoder:
    """Load the frozen effect MLP and codebook from an artifact-version-3 checkpoint."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Effect-tokenizer checkpoint does not exist: {resolved_path}")

    checkpoint = _load_trusted_checkpoint(resolved_path)
    _validate_checkpoint_contract(checkpoint, resolved_path)

    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    saved_config = checkpoint.get("config")
    if not isinstance(model_config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Effect-tokenizer checkpoint {resolved_path} must contain model_config and model_state_dict mappings."
        )
    if not isinstance(saved_config, Mapping) or not isinstance(saved_config.get("data"), Mapping):
        raise ValueError(
            f"Effect-tokenizer checkpoint {resolved_path} is missing the config.data contract."
        )
    data_config = saved_config["data"]

    normalization = data_config.get("action_normalization")
    if normalization != EFFECT_NORMALIZATION_CONTRACT:
        raise ValueError(
            f"Effect-tokenizer checkpoint {resolved_path} uses action_normalization={normalization!r}; "
            f"expected {EFFECT_NORMALIZATION_CONTRACT!r}."
        )

    input_dim = int(model_config.get("input_dim", len(EFFECT_DESCRIPTOR_NAMES)))
    hidden_dim = int(model_config.get("hidden_dim", 128))
    latent_dim = int(model_config.get("latent_dim", 16))
    num_hidden_layers = int(model_config.get("num_hidden_layers", 2))
    codebook_size = int(model_config.get("codebook_size", 256))
    normalize_latents = bool(model_config.get("normalize_latents", False))
    if input_dim != len(EFFECT_DESCRIPTOR_NAMES):
        raise ValueError(f"Effect-tokenizer input_dim must be 7, got {input_dim} in {resolved_path}.")

    encoder = _make_mlp(input_dim, latent_dim, hidden_dim, num_hidden_layers)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder weights found in effect-tokenizer checkpoint {resolved_path}.")
    encoder.load_state_dict(encoder_state, strict=True)

    codebook = state_dict.get("codebook.weight")
    if not isinstance(codebook, Tensor):
        raise ValueError(f"Effect-tokenizer checkpoint {resolved_path} is missing codebook.weight.")
    if codebook.shape != (codebook_size, latent_dim):
        raise ValueError(
            f"Unexpected effect-tokenizer codebook shape {tuple(codebook.shape)} in {resolved_path}; "
            f"expected {(codebook_size, latent_dim)}."
        )

    target_control_hz_raw = data_config.get("target_control_hz")
    target_control_hz = None if target_control_hz_raw is None else float(target_control_hz_raw)
    effect_scale = torch.as_tensor(
        checkpoint.get("effect_scale", [1.0] * len(EFFECT_DESCRIPTOR_NAMES)), dtype=torch.float32
    )
    model = EffectVQVAEActionEncoder(
        encoder=encoder,
        codebook=codebook,
        horizon=int(data_config["horizon"]),
        action_dim=int(data_config["action_dim"]),
        target_control_hz=target_control_hz,
        gripper_weight=float(checkpoint["gripper_weight"]),
        effect_scale=effect_scale,
        normalize_latents=normalize_latents,
    )
    model.requires_grad_(False)
    model.eval()
    return model
