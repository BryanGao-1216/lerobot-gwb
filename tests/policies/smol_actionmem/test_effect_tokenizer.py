from __future__ import annotations

import torch
from torch import nn

from lerobot.policies.smol_actionmem.effect_tokenizer import (
    EFFECT_DESCRIPTOR_NAMES,
    EFFECT_NORMALIZATION_CONTRACT,
    EffectVQVAEActionEncoder,
    compute_effect_descriptors,
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
        horizon=2,
        action_dim=7,
        target_control_hz=10.0,
        gripper_weight=2.0,
        effect_scale=torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        normalize_latents=True,
    )
    actions = torch.zeros(2, 2, 7)
    actions[0, :, 0] = 1.0
    actions[1, -1, 6] = 1.0

    distances = model.compute_code_distances(actions)

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
    checkpoint = tmp_path / "effect_tokenizer.pt"
    state_dict = {f"encoder.{key}": value for key, value in encoder.state_dict().items()}
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
                    "horizon": 10,
                    "action_dim": 7,
                    "target_control_hz": 10.0,
                    "action_normalization": EFFECT_NORMALIZATION_CONTRACT,
                }
            },
        },
        checkpoint,
    )

    loaded = load_effect_vqvae_action_encoder(checkpoint)
    metadata = load_effect_tokenizer_metadata(checkpoint)

    assert loaded.horizon == 10
    assert loaded.action_dim == 7
    assert loaded.codebook_size == 4
    assert loaded.target_control_hz == 10.0
    assert metadata.horizon == 10
    assert metadata.codebook_size == 4
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.compute_code_distances(torch.zeros(2, 10, 7)).shape == (2, 4)
