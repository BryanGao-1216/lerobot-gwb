import json

import pytest
import torch

from lerobot.policies.smol_actionmem.processor_smol_actionmem import (
    SmolActionMemActionTokenProcessorStep,
)
from lerobot.policies.smol_actionmem.tokenization_smol_actionmem import (
    SmolActionMemTokenMap,
    action_token_strings,
)
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_TOKENS,
)


def _write_token_map(tmp_path):
    path = tmp_path / "token_map.json"
    path.write_text(
        json.dumps(
            {
                "vqvae": {
                    "codebook_size": 256,
                    "code_id_min": 0,
                    "code_id_max": 255,
                    "invalid_value": -1,
                    "action_horizon": 16,
                    "action_dim": 7,
                },
                "action_tokens": {
                    "anchor_token_id": 355,
                    "token_id_min": 100,
                    "token_id_max": 355,
                },
                "control_tokens": {
                    "action_memory_start": {"token_id": 356},
                    "action_memory_end": {"token_id": 357},
                    "action_query": {"token_id": 358},
                },
                "padding": {"token_id": 2},
            }
        ),
        encoding="utf-8",
    )
    return path


def _transition(q0=None, batch_size=2):
    complementary_data = {"task": ["pick"] * batch_size}
    if q0 is not None:
        complementary_data[ACTION_TOKEN] = q0
    return create_transition(
        observation={OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 4, dtype=torch.long)},
        complementary_data=complementary_data,
    )


def test_action_token_strings_follow_reverse_mapping():
    strings = action_token_strings(3)
    assert strings[:3] == ["<|action_002|>", "<|action_001|>", "<|action_000|>"]
    assert strings[-3:] == [
        "<|action_memory_start|>",
        "<|action_memory_end|>",
        "<|action_query|>",
    ]


def test_token_map_reports_required_vocabulary_size(tmp_path):
    token_map = SmolActionMemTokenMap.from_json(_write_token_map(tmp_path))
    assert token_map.q0_to_token_id(0) == 355
    assert token_map.q0_to_token_id(255) == 100
    assert token_map.token_id_to_q0(100) == 255
    assert token_map.required_vocab_size == 359


def test_processor_maps_q0_and_reserves_inference_target(tmp_path):
    step = SmolActionMemActionTokenProcessorStep(str(_write_token_map(tmp_path)))
    output = step(_transition(torch.tensor([[0], [255]])))
    data = output[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(
        data[ACTION_TOKENS],
        torch.tensor([[356, 357, 358, 355], [356, 357, 358, 100]]),
    )
    assert data[ACTION_TOKEN_MASK].all()

    inference = step(_transition())
    inference_data = inference[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(
        inference_data[ACTION_TOKENS],
        torch.tensor([[356, 357, 358, 2], [356, 357, 358, 2]]),
    )
    assert torch.equal(
        inference_data[ACTION_TOKEN_MASK],
        torch.tensor([[True, True, True, False], [True, True, True, False]]),
    )


def test_processor_rejects_out_of_range_q0(tmp_path):
    step = SmolActionMemActionTokenProcessorStep(str(_write_token_map(tmp_path)))
    with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
        step(_transition(torch.tensor([[256], [0]])))
