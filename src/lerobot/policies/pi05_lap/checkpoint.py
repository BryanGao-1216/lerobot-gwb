"""Strict native PI05 -> LAP and LAP -> LAP checkpoint loading."""

import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

EMBEDDING_KEY = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
HEAD_KEY = "model.paligemma_with_expert.paligemma.lm_head.weight"
VISION_PREFIX = "model.paligemma_with_expert.paligemma.model.vision_tower."

logger = logging.getLogger(__name__)


def _remap_vision_tower_keys(state: dict, expected: dict) -> dict:
    """Remove a legacy SigLIP wrapper only where the current model expects its contents.

    Some PI05 checkpoints saved SiglipVisionModel's `vision_model` submodule,
    while the current PaliGemma registers that submodule directly as vision_tower.
    Match actual destination keys instead of applying a global string replacement.
    """
    legacy_prefix = VISION_PREFIX + "vision_model."
    remapped = {}
    sources = {}
    count = 0
    for key, value in state.items():
        target = key
        if key.startswith(legacy_prefix) and key not in expected:
            candidate = VISION_PREFIX + key.removeprefix(legacy_prefix)
            if candidate in expected:
                target = candidate
                count += 1
        if target in remapped:
            raise ValueError(
                f"Checkpoint keys {sources[target]!r} and {key!r} both map to {target!r}; "
                "refusing to overwrite a weight."
            )
        remapped[target] = value
        sources[target] = key
    if count:
        logger.info("Mapped %d legacy PI05 vision_tower.vision_model keys to vision_tower.", count)
    return remapped


def checkpoint_file(source, filename: str, **hub_args) -> Path:
    path = Path(source)
    if path.is_dir():
        path = path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file missing: {path}")
        return path
    return Path(hf_hub_download(str(source), filename, **hub_args))


def checkpoint_type(source, **hub_args) -> str:
    value = json.loads(checkpoint_file(source, "config.json", **hub_args).read_text()).get("type")
    if value not in {"pi05", "pi05_lap"}:
        raise ValueError(f"Expected a PI05 or PI05 LAP checkpoint, got {value!r}.")
    return value


def load_student_state(policy, source, *, source_type: str, **hub_args):
    from safetensors.torch import load_file

    state = load_file(str(checkpoint_file(source, "model.safetensors", **hub_args)))
    if source_type == "pi05":
        state = policy._fix_pytorch_state_dict_keys(state, policy.config)
    state = {k if k.startswith("model.") else f"model.{k}": v for k, v in state.items()}
    expected = policy.state_dict()
    state = _remap_vision_tower_keys(state, expected)
    # safetensors omits one copy of a tied input/output tensor. Only this known alias
    # may be absent; all other missing or unexpected tensors are fatal.
    vlm = policy.model.paligemma_with_expert.paligemma
    if source_type == "pi05" or vlm.get_input_embeddings().weight is vlm.get_output_embeddings().weight:
        for key, alias in ((EMBEDDING_KEY, HEAD_KEY), (HEAD_KEY, EMBEDDING_KEY)):
            if key not in state and alias in state:
                state[key] = state[alias]
    if source_type == "pi05":
        for key in (EMBEDDING_KEY, HEAD_KEY):
            source_weight = state.get(key)
            target_weight = expected[key]
            if source_weight is None:
                continue  # Strict load below reports the missing tensor.
            if source_weight.shape != target_weight.shape:
                if source_weight.shape != (257152, target_weight.shape[1]) or len(target_weight) < 257152:
                    raise ValueError(
                        f"Unexpected native PI05 vocabulary shape for {key}: {source_weight.shape}"
                    )
                # Only the newly introduced token rows retain their initialization.
                expanded = target_weight.detach().clone()
                expanded[: len(source_weight)].copy_(source_weight)
                state[key] = expanded
    policy.load_state_dict(state, strict=True)
