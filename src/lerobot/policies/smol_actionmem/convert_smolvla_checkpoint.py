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

"""Convert a pretrained SmolVLA artifact into a Smol ActionMem base artifact.

Example:

python -m lerobot.policies.smol_actionmem.convert_smolvla_checkpoint \
  --source lerobot/smolvla_base \
  --output-dir /path/to/models/smol_actionmem-base \
  --source-token-map /path/to/actionmem_token_map.json \
  --vqvae-checkpoint /path/to/action_vqvae.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import fields
from pathlib import Path

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smol_actionmem import SmolActionMemConfig
from .modeling_smol_actionmem import SmolActionMemPolicy
from .processor_smol_actionmem import make_smol_actionmem_pre_post_processors
from .tokenization_smol_actionmem import select_low_frequency_token_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-token-map", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path)
    parser.add_argument("--tokenizer-source")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in an existing output directory.",
    )
    return parser.parse_args()


def _prepare_output_directory(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Pass --overwrite to replace generated files."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_tokenizer_and_map(
    *,
    tokenizer_source: str,
    source_token_map_path: Path,
    output_dir: Path,
    vqvae_checkpoint: Path | None,
) -> Path:
    from transformers import AutoProcessor

    with source_token_map_path.expanduser().resolve().open(encoding="utf-8") as file:
        source_map = json.load(file)
    vqvae = copy.deepcopy(source_map["vqvae"])
    codebook_size = int(vqvae["codebook_size"])

    processor = AutoProcessor.from_pretrained(tokenizer_source)
    tokenizer = processor.tokenizer
    action_ids, control_ids = select_low_frequency_token_ids(tokenizer, codebook_size)
    action_strings = [str(tokenizer.convert_ids_to_tokens(token_id)) for token_id in action_ids]
    expected_ids = list(range(min(action_ids), min(action_ids) + codebook_size))
    if action_ids != expected_ids:
        raise ValueError(
            "Action token IDs must form one existing contiguous range; got "
            f"{action_ids[:3]} ... {action_ids[-3:]}."
        )

    control_strings = [str(tokenizer.convert_ids_to_tokens(token_id)) for token_id in control_ids]
    if len(set(action_ids + control_ids)) != codebook_size + 3:
        raise ValueError("The tokenizer did not assign unique IDs to all Smol ActionMem tokens.")

    if vqvae_checkpoint is not None:
        vqvae["checkpoint_path"] = str(vqvae_checkpoint.expanduser().resolve())

    # q0 codes use the same reverse-contiguous formula as ActionMem: the
    # highest selected action ID represents q0=0.
    token_map = {
        "format_version": 1,
        "name": "smol_actionmem_q0_low_frequency_token_map",
        "description": (
            "Maps the first residual VQ codebook to existing high-rank, low-frequency "
            "tokens in the SmolVLM base BPE vocabulary."
        ),
        "vqvae": vqvae,
        "smolvlm": {
            "tokenizer_source": tokenizer_source,
            "base_vocab_size": int(tokenizer.vocab_size),
            "tokenizer_length": len(tokenizer),
            "vocabulary_strategy": "reuse_high_rank_base_bpe_tokens",
            "reused_token_id_min": min(action_ids),
            "reused_token_id_max": max(control_ids),
        },
        "action_tokens": {
            "count": codebook_size,
            "mapping_type": "reverse_contiguous",
            "mapping_formula": "token_id = anchor_token_id - q0_code_id",
            "anchor_token_id": max(action_ids),
            "token_id_min": min(action_ids),
            "token_id_max": max(action_ids),
            "source_token_strings_by_ascending_id": action_strings,
        },
        "control_tokens": {
            "action_memory_start": {
                "token_id": control_ids[0],
                "source_token": control_strings[0],
            },
            "action_memory_end": {
                "token_id": control_ids[1],
                "source_token": control_strings[1],
            },
            "action_query": {
                "token_id": control_ids[2],
                "source_token": control_strings[2],
            },
        },
        "padding": {
            "token_id": int(tokenizer.pad_token_id),
            "token": tokenizer.pad_token,
        },
    }
    token_map_path = output_dir / "token_map.json"
    with token_map_path.open("w", encoding="utf-8") as file:
        json.dump(token_map, file, indent=2, ensure_ascii=False)
        file.write("\n")
    processor.save_pretrained(output_dir)
    return token_map_path


def _make_target_config(
    source_policy: SmolVLAPolicy,
    *,
    token_map_path: Path,
    chunk_size: int,
    n_action_steps: int,
) -> SmolActionMemConfig:
    target_field_names = {field.name for field in fields(SmolActionMemConfig)}
    source_config = source_policy.config
    values = {
        name: copy.deepcopy(getattr(source_config, name))
        for name in target_field_names
        if hasattr(source_config, name)
    }
    values.update(
        {
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "num_inference_steps": int(getattr(source_config, "num_steps", 10)),
            "action_token_map_path": str(token_map_path),
            "action_vqvae_checkpoint_path": None,
            "action_vqvae_flow_mean": None,
            "action_vqvae_flow_std": None,
            "training_stage": "joint",
            "train_expert_only": False,
            "gradient_checkpointing": False,
            "pretrained_path": None,
            "repo_id": None,
            "push_to_hub": False,
        }
    )
    return SmolActionMemConfig(**values)


def _copy_compatible_weights(source_policy: SmolVLAPolicy, target_policy: SmolActionMemPolicy) -> None:
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    compatible_state = {}
    expanded_keys = []
    skipped_keys = []

    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped_keys.append(key)
            continue
        if source_value.shape == target_value.shape:
            compatible_state[key] = source_value
            continue

        is_vocab_matrix = (
            source_value.ndim == target_value.ndim == 2
            and source_value.shape[1:] == target_value.shape[1:]
            and source_value.shape[0] < target_value.shape[0]
            and (key.endswith("text_model.embed_tokens.weight") or key.endswith("lm_head.weight"))
        )
        if is_vocab_matrix:
            expanded = target_value.detach().clone()
            expanded[: source_value.shape[0]].copy_(
                source_value.to(device=expanded.device, dtype=expanded.dtype)
            )
            compatible_state[key] = expanded
            expanded_keys.append(key)
        else:
            skipped_keys.append(key)

    missing_keys, unexpected_keys = target_policy.load_state_dict(compatible_state, strict=False)
    logging.info(
        "Transferred %d tensors; expanded %d vocabulary tensors; skipped %d source tensors.",
        len(compatible_state),
        len(expanded_keys),
        len(skipped_keys),
    )
    if unexpected_keys:
        raise RuntimeError(f"Unexpected converted checkpoint keys: {unexpected_keys}")
    actionmem_only_missing = [
        key
        for key in missing_keys
        if key not in expanded_keys
        and not key.endswith(
            (
                "text_model.embed_tokens.weight",
                "lm_head.weight",
            )
        )
    ]
    if actionmem_only_missing:
        logging.warning(
            "Target-only or unmatched tensors retain their initialization: %s",
            actionmem_only_missing,
        )


def convert(args: argparse.Namespace) -> Path:
    output_dir = _prepare_output_directory(args.output_dir, args.overwrite)
    logging.info("Loading source SmolVLA policy from %s", args.source)
    source_policy = SmolVLAPolicy.from_pretrained(args.source)
    tokenizer_source = args.tokenizer_source or source_policy.config.vlm_model_name
    token_map_path = _create_tokenizer_and_map(
        tokenizer_source=tokenizer_source,
        source_token_map_path=args.source_token_map,
        output_dir=output_dir,
        vqvae_checkpoint=args.vqvae_checkpoint,
    )
    target_config = _make_target_config(
        source_policy,
        token_map_path=token_map_path,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
    )
    target_policy = SmolActionMemPolicy(target_config)
    _copy_compatible_weights(source_policy, target_policy)

    target_policy.save_pretrained(output_dir)
    preprocessor, postprocessor = make_smol_actionmem_pre_post_processors(target_config)
    preprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )
    logging.info("Saved Smol ActionMem base artifact to %s", output_dir)
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    with torch.no_grad():
        convert(_parse_args())


if __name__ == "__main__":
    main()
