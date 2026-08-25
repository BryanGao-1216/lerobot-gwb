import pytest
import torch

from lerobot.policies.actionmem.processor_actionmem import ActionMemActionTokenProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION_TOKEN, ACTION_TOKEN_MASK, ACTION_TOKENS, OBS_LANGUAGE_TOKENS


def _make_transition(codes=None, batch_size=2):
    complementary_data = {"task": ["pick"] * batch_size}
    if codes is not None:
        complementary_data[ACTION_TOKEN] = codes
    return create_transition(
        observation={OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 4, dtype=torch.long)},
        complementary_data=complementary_data,
    )


def test_actionmem_action_code_processor_keeps_local_code_ids():
    step = ActionMemActionTokenProcessorStep(codebook_size=256, invalid_value=-1)
    output = step(_make_transition(torch.tensor([[0], [255]])))[TransitionKey.COMPLEMENTARY_DATA]

    assert torch.equal(
        output[ACTION_TOKENS],
        torch.tensor([[256, 257, 258, 0], [256, 257, 258, 255]]),
    )
    assert output[ACTION_TOKEN_MASK].all()


def test_actionmem_action_code_processor_masks_invalid_and_inference_targets():
    step = ActionMemActionTokenProcessorStep(codebook_size=256, invalid_value=-1)

    invalid = step(_make_transition(torch.tensor([[-1], [42]])))[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(
        invalid[ACTION_TOKENS],
        torch.tensor([[256, 257, 258, 259], [256, 257, 258, 42]]),
    )
    assert torch.equal(
        invalid[ACTION_TOKEN_MASK],
        torch.tensor([[True, True, True, False], [True, True, True, True]]),
    )

    inference = step(_make_transition())[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(
        inference[ACTION_TOKENS],
        torch.tensor([[256, 257, 258, 259], [256, 257, 258, 259]]),
    )
    assert not inference[ACTION_TOKEN_MASK][:, -1].any()


def test_actionmem_action_code_processor_rejects_out_of_range_codes():
    step = ActionMemActionTokenProcessorStep(codebook_size=256, invalid_value=-1)

    with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
        step(_make_transition(torch.tensor([[256], [0]])))
