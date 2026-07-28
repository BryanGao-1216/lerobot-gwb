import json

import pytest
import torch

from lerobot.policies.actionmem.processor_actionmem import ActionMemActionTokenProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_TOKENS,
)


def _write_token_map(tmp_path):
    token_map_path = tmp_path / "actionmem_token_map.json"
    token_map_path.write_text(
        json.dumps(
            {
                "vqvae": {
                    "codebook_size": 256,
                    "code_id_min": 0,
                    "code_id_max": 255,
                    "invalid_value": -1,
                },
                "action_tokens": {
                    "anchor_token_id": 257023,
                    "token_id_min": 256768,
                    "token_id_max": 257023,
                },
                "control_tokens": {"action_query": {"token_id": 9}},
                "padding": {"token_id": 0},
            }
        ),
        encoding="utf-8",
    )
    return token_map_path


def _make_transition(q0=None, batch_size=2):
    complementary_data = {"task": ["pick"] * batch_size}
    if q0 is not None:
        complementary_data[ACTION_TOKEN] = q0
    return create_transition(
        observation={OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 4, dtype=torch.long)},
        complementary_data=complementary_data,
    )


def test_actionmem_action_token_processor_maps_q0(tmp_path):
    step = ActionMemActionTokenProcessorStep(token_map_path=str(_write_token_map(tmp_path)))
    transition = _make_transition(torch.tensor([[0], [255]], dtype=torch.long))

    output = step(transition)
    complementary_data = output[TransitionKey.COMPLEMENTARY_DATA]

    assert torch.equal(
        complementary_data[ACTION_TOKENS],
        torch.tensor([[9, 257023], [9, 256768]], dtype=torch.long),
    )
    assert torch.equal(
        complementary_data[ACTION_TOKEN_MASK],
        torch.tensor([[True, True], [True, True]]),
    )


def test_actionmem_action_token_processor_masks_invalid_and_inference_targets(tmp_path):
    step = ActionMemActionTokenProcessorStep(token_map_path=str(_write_token_map(tmp_path)))

    invalid_output = step(_make_transition(torch.tensor([[-1], [42]], dtype=torch.long)))
    invalid_data = invalid_output[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(
        invalid_data[ACTION_TOKENS],
        torch.tensor([[9, 0], [9, 256981]], dtype=torch.long),
    )
    assert torch.equal(
        invalid_data[ACTION_TOKEN_MASK],
        torch.tensor([[True, False], [True, True]]),
    )

    inference_output = step(_make_transition(q0=None))
    inference_data = inference_output[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(
        inference_data[ACTION_TOKENS],
        torch.tensor([[9, 0], [9, 0]], dtype=torch.long),
    )
    assert torch.equal(
        inference_data[ACTION_TOKEN_MASK],
        torch.tensor([[True, False], [True, False]]),
    )


def test_actionmem_action_token_processor_rejects_out_of_range_q0(tmp_path):
    step = ActionMemActionTokenProcessorStep(token_map_path=str(_write_token_map(tmp_path)))

    with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
        step(_make_transition(torch.tensor([[256], [0]], dtype=torch.long)))
