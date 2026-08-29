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
  --effect-tokenizer-checkpoint /path/to/effect_vqvae.pt \
  --chunk-size 20
"""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import fields
from pathlib import Path

import torch

from lerobot.policies.effect_tokenizer import EffectTokenizerMetadata, load_effect_tokenizer_metadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smol_actionmem import SmolActionMemConfig
from .modeling_smol_actionmem import SmolActionMemPolicy
from .processor_smol_actionmem import make_smol_actionmem_pre_post_processors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--effect-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--tokenizer-source")
    parser.add_argument("--n-action-steps", type=int)
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


def _copy_tokenizer_assets(*, tokenizer_source: str, output_dir: Path) -> None:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(tokenizer_source)
    processor.save_pretrained(output_dir)


def _make_target_config(
    source_policy: SmolVLAPolicy,
    *,
    effect_metadata: EffectTokenizerMetadata,
    chunk_size: int,
    n_action_steps: int | None,
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
            "n_action_steps": n_action_steps or chunk_size,
            "drop_n_last_frames": 0,
            "num_inference_steps": int(getattr(source_config, "num_steps", 10)),
            "effect_tokenizer_checkpoint_path": effect_metadata.checkpoint_path,
            "action_codebook_size": effect_metadata.codebook_size,
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
        "model.action_condition_proj.",
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
    _copy_tokenizer_assets(tokenizer_source=tokenizer_source, output_dir=output_dir)
    effect_metadata = load_effect_tokenizer_metadata(args.effect_tokenizer_checkpoint)
    target_config = _make_target_config(
        source_policy,
        effect_metadata=effect_metadata,
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
