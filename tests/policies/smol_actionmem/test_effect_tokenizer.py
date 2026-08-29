from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lerobot.policies.effect_tokenizer import (
    EFFECT_DESCRIPTOR_NAMES,
    EFFECT_NORMALIZATION_CONTRACT,
    EffectVQVAEActionEncoder,
    compute_effect_descriptors,
    load_effect_token_prototypes,
    load_effect_tokenizer_metadata,
    load_effect_vqvae_action_encoder,
)


def test_effect_descriptors_accumulate_motion_and_difference_gripper():
    actions = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.25],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.75],
            ]
        ]
    )

    effects = compute_effect_descriptors(actions)

    assert torch.allclose(effects, torch.tensor([[5.0, 7.0, 9.0, 0.5, 0.7, 0.9, 0.5]]))


def test_effect_encoder_applies_checkpoint_scale_before_latent_distance():
    encoder = nn.Linear(7, 2, bias=False)
    encoder.weight.data.zero_()
    encoder.weight.data[0, 0] = 1.0
    encoder.weight.data[1, 6] = 1.0
    model = EffectVQVAEActionEncoder(
        encoder=encoder,
        codebook=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        action_dim=7,
        gripper_weight=2.0,
        effect_scale=torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        normalize_latents=True,
    )
    effects = torch.zeros(2, 7)
    effects[0, 0] = 2.0
    effects[1, 6] = 1.0

    distances = model.compute_code_distances_from_effects(effects)

    assert torch.equal(distances.argmin(dim=-1), torch.tensor([0, 1]))


def test_load_effect_tokenizer_checkpoint_contract(tmp_path):
    hidden_dim = 5
    latent_dim = 3
    model_config = {
        "input_dim": 7,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_hidden_layers": 1,
        "codebook_size": 4,
        "normalize_latents": True,
    }
    encoder = nn.Sequential(nn.Linear(7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
    decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 7))
    checkpoint = tmp_path / "effect_tokenizer.pt"
    state_dict = {f"encoder.{key}": value for key, value in encoder.state_dict().items()}
    state_dict.update({f"decoder.{key}": value for key, value in decoder.state_dict().items()})
    state_dict["codebook.weight"] = torch.randn(4, latent_dim)
    torch.save(
        {
            "artifact_version": 3,
            "model_type": "mlp_effect_vqvae",
            "model_config": model_config,
            "model_state_dict": state_dict,
            "gripper_weight": 1.5,
            "effect_scale": [0.1] * 6 + [1.0],
            "descriptor_names": list(EFFECT_DESCRIPTOR_NAMES),
            "config": {
                "data": {
                    "window_contract_version": 2,
                    "window_duration_seconds": 1.0,
                    "sampling_stride_seconds": 0.25,
                    "pad_incomplete_windows": True,
                    "action_dim": 7,
                    "action_normalization": EFFECT_NORMALIZATION_CONTRACT,
                }
            },
        },
        checkpoint,
    )

    loaded = load_effect_vqvae_action_encoder(checkpoint)
    metadata = load_effect_tokenizer_metadata(checkpoint)
    prototypes = load_effect_token_prototypes(checkpoint)
    expected_prototypes = decoder(F.normalize(state_dict["codebook.weight"], dim=-1))
    expected_prototypes[:, -1] /= 1.5
    expected_prototypes /= torch.tensor([0.1] * 6 + [1.0])

    assert loaded.action_dim == 7
    assert loaded.codebook_size == 4
    assert metadata.window_contract_version == 2
    assert metadata.window_duration_seconds == 1.0
    metadata.validate_policy_horizon(10, 10.0)
    with pytest.raises(ValueError, match="chunk_size=20"):
        metadata.validate_policy_horizon(20, 10.0)
    assert metadata.codebook_size == 4
    assert torch.allclose(prototypes, expected_prototypes)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.compute_code_distances_from_effects(torch.zeros(2, 7)).shape == (2, 4)
