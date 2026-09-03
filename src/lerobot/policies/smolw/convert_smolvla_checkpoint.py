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

"""Convert an original SmolVLA artifact into a SmolW base artifact.

Only shape-compatible original SmolVLA tensors are transferred.  The M_t
query, motion prediction head, and action-conditioning projection retain their
SmolW initialization.

Example:

python -m lerobot.policies.smolw.convert_smolvla_checkpoint \
  --source /path/to/models/smolvla \
  --output-dir /path/to/models/smolw-base \
  --vidtwin-checkpoint-path /path/to/vidtwin_structure_7_7_8_dynamics_7_8.ckpt \
  --motion-horizon 16
"""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import fields
from pathlib import Path

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smolw import SmolWConfig
from .modeling_smolw import SmolWPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vidtwin-checkpoint-path", type=Path, required=True)
    parser.add_argument("--motion-horizon", type=int, required=True)
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--motion-camera-key")
    parser.add_argument("--motion-loss-weight", type=float, default=1.0)
    parser.add_argument("--future-visual-loss-weight", type=float, default=1.0)
    parser.add_argument("--future-visual-cosine-weight", type=float, default=0.1)
    parser.add_argument("--motion-latent-dim", type=int, default=1792)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting generated files in a non-empty output directory.",
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


def _make_target_config(source_policy: SmolVLAPolicy, args: argparse.Namespace) -> SmolWConfig:
    target_field_names = {field.name for field in fields(SmolWConfig)}
    source_config = source_policy.config
    values = {
        name: copy.deepcopy(getattr(source_config, name))
        for name in target_field_names
        if hasattr(source_config, name)
    }
    values.update(
        {
            "chunk_size": args.motion_horizon,
            "n_action_steps": args.n_action_steps or args.motion_horizon,
            "motion_horizon": args.motion_horizon,
            "memory_stride": args.memory_stride,
            "drop_n_last_frames": args.motion_horizon,
            "vidtwin_checkpoint_path": str(args.vidtwin_checkpoint_path.expanduser().resolve()),
            "vidtwin_sample_posterior": False,
            "motion_camera_key": args.motion_camera_key,
            "motion_loss_weight": args.motion_loss_weight,
            "future_visual_loss_weight": args.future_visual_loss_weight,
            "future_visual_cosine_weight": args.future_visual_cosine_weight,
            "motion_latent_dim": args.motion_latent_dim,
            "training_stage": "world_model",
            "train_expert_only": False,
            # The converter transfers the complete original SmolVLA state, so
            # the target does not need to download VLM weights a second time.
            "load_vlm_weights": False,
            "compile_model": False,
            "pretrained_path": None,
            "repo_id": None,
            "push_to_hub": False,
        }
    )
    return SmolWConfig(**values)


def _copy_compatible_weights(source_policy: SmolVLAPolicy, target_policy: SmolWPolicy) -> None:
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    compatible_state = {
        key: source_value
        for key, source_value in source_state.items()
        if key in target_state and source_value.shape == target_state[key].shape
    }
    missing_keys, unexpected_keys = target_policy.load_state_dict(compatible_state, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected converted checkpoint keys: {unexpected_keys}")

    expected_new_prefixes = (
        "model.mt_query_embedding.",
        "model.past_motion_projector.",
        "model.future_motion_head.",
        "model.future_motion_condition_proj.",
        "model.future_visual_queries.",
        "model.future_motion_visual_proj.",
        "model.future_visual_decoder.",
        "model.future_visual_out_proj.",
    )
    unmatched_missing = [key for key in missing_keys if not key.startswith(expected_new_prefixes)]
    logging.info(
        "Transferred %d/%d original SmolVLA tensors into SmolW.",
        len(compatible_state),
        len(source_state),
    )
    if unmatched_missing:
        logging.warning("Target tensors retaining their initialization: %s", unmatched_missing)


def convert(args: argparse.Namespace) -> Path:
    output_dir = _prepare_output_directory(args.output_dir, args.overwrite)
    logging.info("Loading original SmolVLA policy from %s", args.source)
    source_policy = SmolVLAPolicy.from_pretrained(args.source)
    target_config = _make_target_config(source_policy, args)
    target_policy = SmolWPolicy(target_config)
    _copy_compatible_weights(source_policy, target_policy)
    target_policy.save_pretrained(output_dir)

    # SmolW deliberately retains the original SmolVLA processor contract.
    preprocessor, postprocessor = make_pre_post_processors(
        source_policy.config,
        pretrained_path=args.source,
    )
    preprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor.save_pretrained(
        output_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )
    logging.info("Saved SmolW base artifact to %s", output_dir)
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    with torch.no_grad():
        convert(_parse_args())


if __name__ == "__main__":
    main()
