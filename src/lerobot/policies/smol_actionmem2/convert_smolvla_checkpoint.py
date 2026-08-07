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

"""Convert a pretrained SmolVLA artifact into a Smol ActionMem 2 base artifact.

Example:

python -m lerobot.policies.smol_actionmem2.convert_smolvla_checkpoint \
  --source lerobot/smolvla_base \
  --output-dir /path/to/models/smol_actionmem2-base \
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

from .configuration_smol_actionmem import SmolActionMem2Config
from .modeling_smol_actionmem import SmolActionMem2Policy
from .processor_smol_actionmem import make_smol_actionmem2_pre_post_processors


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
    processor.save_pretrained(output_dir)

    if vqvae_checkpoint is not None:
        vqvae["checkpoint_path"] = str(vqvae_checkpoint.expanduser().resolve())

    token_map = {
        "format_version": 2,
        "name": "smol_actionmem2_q0_class_map",
        "description": (
            "Describes the first residual VQ codebook used by Smol ActionMem 2's "
            "independent classifier and embedding table."
        ),
        "vqvae": vqvae,
        "smolvlm": {
            "tokenizer_source": tokenizer_source,
            "vocabulary_strategy": "independent_action_vocabulary",
            "language_vocabulary_modified": False,
        },
        "action_classes": {
            "count": codebook_size,
            "class_id_min": 0,
            "class_id_max": codebook_size - 1,
            "mapping_formula": "class_id = q0_code_id - vqvae.code_id_min",
        },
        "action_context": {
            "embedding_size": codebook_size + 4,
            "memory_start_id": codebook_size,
            "memory_end_id": codebook_size + 1,
            "action_query_id": codebook_size + 2,
            "padding_id": codebook_size + 3,
        },
    }
    token_map_path = output_dir / "token_map.json"
    with token_map_path.open("w", encoding="utf-8") as file:
        json.dump(token_map, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return token_map_path


def _make_target_config(
    source_policy: SmolVLAPolicy,
    *,
    token_map_path: Path,
    chunk_size: int,
    n_action_steps: int,
) -> SmolActionMem2Config:
    target_field_names = {field.name for field in fields(SmolActionMem2Config)}
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
            "action_vqvae_input_q01": None,
            "action_vqvae_input_q99": None,
            "action_vqvae_input_mask": None,
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
    return SmolActionMem2Config(**values)


def _copy_compatible_weights(source_policy: SmolVLAPolicy, target_policy: SmolActionMem2Policy) -> None:
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped_keys.append(key)
            continue
        if source_value.shape == target_value.shape:
            compatible_state[key] = source_value
            continue
        skipped_keys.append(key)

    missing_keys, unexpected_keys = target_policy.load_state_dict(compatible_state, strict=False)
    logging.info(
        "Transferred %d tensors without changing the language vocabulary; skipped %d source tensors.",
        len(compatible_state),
        len(skipped_keys),
    )
    if unexpected_keys:
        raise RuntimeError(f"Unexpected converted checkpoint keys: {unexpected_keys}")
    expected_new_modules = (
        "model.action_code_embedding.",
        "model.action_classifier.",
    )
    expected_missing = [key for key in missing_keys if key.startswith(expected_new_modules)]
    unmatched_missing = [key for key in missing_keys if not key.startswith(expected_new_modules)]
    if expected_missing:
        logging.info("Initialized new independent action modules: %s", expected_missing)
    if unmatched_missing:
        logging.warning(
            "Target-only or unmatched tensors retain their initialization: %s",
            unmatched_missing,
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
    target_policy = SmolActionMem2Policy(target_config)
    _copy_compatible_weights(source_policy, target_policy)

    target_policy.save_pretrained(output_dir)
    preprocessor, postprocessor = make_smol_actionmem2_pre_post_processors(target_config)
    preprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )
    logging.info("Saved Smol ActionMem 2 base artifact to %s", output_dir)
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    with torch.no_grad():
        convert(_parse_args())


if __name__ == "__main__":
    main()
