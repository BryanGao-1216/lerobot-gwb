#!/usr/bin/env python

"""Visualize Smol ActionMem Top-K action-token predictions on random OXE samples."""

from __future__ import annotations

import argparse
import json
import logging
import random
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import DatasetConfig, PreTrainedConfig
from lerobot.datasets.rlds_dataset import ActionMemRLDSDataset
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.effect_tokenizer import (
    load_effect_token_prototypes,
    load_effect_tokenizer_metadata,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smol_actionmem.configuration_smol_actionmem import (
    SmolActionMemConfig,
)
from lerobot.policies.smol_actionmem.modeling_smol_actionmem import (
    SmolActionMemPolicy,
)
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENIZER_INPUT,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

_REPORT_TEMPLATE = Path(__file__).with_name("report_template.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample RLDS/OXE observations, predict Smol ActionMem action-token probabilities, "
            "and draw the Top-K decoded endpoint prototypes beside the true action trajectory."
        )
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--effect-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--mixture-path", type=Path)
    parser.add_argument(
        "--rlds-storage-format",
        choices=("auto", "tfds", "webdataset", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--target-control-hz", type=float, default=10.0)
    parser.add_argument("--camera-views", default="primary,secondary,wrist")
    parser.add_argument("--resize-height", type=int, default=256)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--shuffle-buffer-size", type=int, default=4096)
    parser.add_argument("--num-parallel-calls", type=int, default=8)
    parser.add_argument("--state-dim", type=int)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/action_token_eval"))
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.target_control_hz < 0:
        parser.error("--target-control-hz must be non-negative")
    if args.resize_height <= 0 or args.resize_width <= 0:
        parser.error("--resize-height and --resize-width must be positive")
    return args


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_single_sample(
    dataset: ActionMemRLDSDataset,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Apply the same RLDS collator used by lerobot-train with batch size one."""
    batch = dataset.collate_fn([sample])
    for camera_key in dataset.meta.camera_keys:
        mask_key = f"{camera_key}_padding_mask"
        mask = batch.get(mask_key)
        if not isinstance(mask, torch.Tensor) or mask.shape != (1,):
            actual = None if not isinstance(mask, torch.Tensor) else tuple(mask.shape)
            raise ValueError(f"Expected batched image mask '{mask_key}' with shape (1,), got {actual}.")
    return batch


def _make_dataset(
    args: argparse.Namespace,
    policy_config: SmolActionMemConfig,
    seed: int,
) -> ActionMemRLDSDataset:
    metadata = load_effect_tokenizer_metadata(args.effect_tokenizer_checkpoint)
    if policy_config.action_codebook_size != metadata.codebook_size:
        raise ValueError(
            f"Policy codebook_size={policy_config.action_codebook_size} does not match "
            f"effect-tokenizer codebook_size={metadata.codebook_size}."
        )
    target_hz = args.target_control_hz if args.target_control_hz > 0 else None
    metadata.validate_policy_horizon(policy_config.chunk_size, target_hz)

    camera_views = tuple(view.strip() for view in args.camera_views.split(",") if view.strip())
    dataset_config = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(args.dataset_root.expanduser().resolve()),
        rlds_mixture_path=None if args.mixture_path is None else str(args.mixture_path.expanduser().resolve()),
        rlds_storage_format=args.rlds_storage_format,
        rlds_target_control_hz=args.target_control_hz,
        rlds_camera_views=camera_views,
        rlds_resize_size=(args.resize_height, args.resize_width),
        rlds_shuffle_buffer_size=args.shuffle_buffer_size,
        rlds_num_parallel_calls=args.num_parallel_calls,
        rlds_action_tokenizer_device="cpu",
        rlds_state_dim=args.state_dim,
    )
    state_dim = args.state_dim or policy_config.max_state_dim
    return ActionMemRLDSDataset(
        dataset_config=dataset_config,
        action_horizon=policy_config.chunk_size,
        action_dim=metadata.action_dim,
        state_dim=state_dim,
        action_tokenizer_checkpoint_path=args.effect_tokenizer_checkpoint,
        action_codebook_size=metadata.codebook_size,
        action_tokenizer_window_duration_seconds=metadata.window_duration_seconds,
        seed=seed,
    )


@torch.inference_mode()
def _predict_action_logits(policy: SmolActionMemPolicy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run only the image/language/state prefix through ACTION_QUERY."""
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    action_tokens = batch[ACTION_TOKENS]
    action_token_masks = batch[ACTION_TOKEN_MASK]

    # The final slot is the supervision-only current code during training and
    # padding during inference. It must never be visible to ACTION_QUERY.
    prompt_tokens = action_tokens[:, :-1]
    prompt_masks = action_token_masks[:, :-1]
    prefix_embeddings, prefix_padding_masks, prefix_attention_blocks = policy.model.embed_prefix(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        prompt_tokens,
        prompt_masks,
        state=state,
    )
    attention = make_att_2d_masks(prefix_padding_masks, prefix_attention_blocks)
    position_ids = torch.cumsum(prefix_padding_masks, dim=1) - 1
    (prefix_output, _), _ = policy.model.vlm_with_expert.forward(
        attention_mask=attention,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, None],
        use_cache=False,
        fill_kv_cache=True,
    )
    return policy.model._compute_action_logits(prefix_output).float()  # noqa: SLF001


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_interactive_html(output_path: Path, report: dict[str, Any]) -> None:
    if not _REPORT_TEMPLATE.is_file():
        raise FileNotFoundError(f"Interactive report template does not exist: {_REPORT_TEMPLATE}")
    template = _REPORT_TEMPLATE.read_text(encoding="utf-8")
    marker = "__ACTION_TOKEN_EVAL_DATA__"
    if template.count(marker) != 1:
        raise ValueError(f"Expected exactly one {marker!r} marker in {_REPORT_TEMPLATE}.")
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), default=_json_value)
    # Prevent user-provided task text from terminating the JSON script element.
    payload = payload.replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace(marker, payload), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    _set_seed(seed)
    device = _resolve_device(args.device)

    policy_path = args.policy_path.expanduser().resolve()
    tokenizer_path = args.effect_tokenizer_checkpoint.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy directory does not exist: {policy_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Effect-tokenizer checkpoint does not exist: {tokenizer_path}")

    loaded_config = PreTrainedConfig.from_pretrained(policy_path)
    if not isinstance(loaded_config, SmolActionMemConfig):
        raise TypeError(
            f"Expected a smol_actionmem checkpoint, got policy type {loaded_config.type!r} from {policy_path}."
        )
    loaded_config.pretrained_path = policy_path
    loaded_config.device = str(device)
    loaded_config.effect_tokenizer_checkpoint_path = str(tokenizer_path)

    logging.info("Building random RLDS/OXE stream with seed=%d", seed)
    dataset = _make_dataset(args, loaded_config, seed)
    logging.info("Loading Smol ActionMem policy from %s", policy_path)
    policy = make_policy(cfg=loaded_config, ds_meta=dataset.meta)
    if not isinstance(policy, SmolActionMemPolicy):
        raise TypeError(f"Loaded policy is {type(policy).__name__}, expected SmolActionMemPolicy.")
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=loaded_config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    all_prototypes = load_effect_token_prototypes(tokenizer_path).numpy()
    if args.top_k > all_prototypes.shape[0]:
        raise ValueError(
            f"top_k={args.top_k} exceeds codebook_size={all_prototypes.shape[0]}."
        )

    run_dir = args.output_dir.expanduser().resolve() / (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    iterator = iter(dataset)
    for sample_index in range(1, args.num_samples + 1):
        sample = next(iterator)
        true_action_chunk = sample[ACTION_TOKENIZER_INPUT].detach().cpu().numpy().astype(np.float32)
        task = str(sample.get("task", ""))
        dataset_name = str(sample.get("dataset_name", "unknown"))
        batch = _collate_single_sample(dataset, sample)

        # Match lerobot-train's collation and uint8-to-[0, 1] conversion before
        # the policy processor. In particular, image padding masks must be [B],
        # not scalars from a raw iterable-dataset sample.
        for camera_key in dataset.meta.camera_keys:
            image = batch.get(camera_key)
            if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
                batch[camera_key] = image.to(torch.float32) / 255.0
        processed_batch = preprocessor(batch)
        logits = _predict_action_logits(policy, processed_batch)
        probabilities = torch.softmax(logits[0], dim=-1)
        top_probabilities, top_tokens = torch.topk(probabilities, k=args.top_k)
        token_ids = top_tokens.detach().cpu().numpy()
        token_probabilities = top_probabilities.detach().cpu().numpy()
        prototypes = all_prototypes[token_ids]

        results.append(
            {
                "sample": sample_index,
                "dataset_name": dataset_name,
                "task": task,
                "top_tokens": token_ids,
                "top_probabilities": token_probabilities,
                "decoded_effect_prototypes": prototypes,
                "true_action_chunk": true_action_chunk,
            }
        )
        logging.info(
            "[%d/%d] collected sample | Top-1 token=%d p=%.4f",
            sample_index,
            args.num_samples,
            int(token_ids[0]),
            float(token_probabilities[0]),
        )

    summary = {
        "seed": seed,
        "policy_path": str(policy_path),
        "effect_tokenizer_checkpoint": str(tokenizer_path),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "coordinate_system": "per-dataset q01/q99-normalized OXE endpoint effects",
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "samples": results,
    }
    report_path = run_dir / "action_token_eval.html"
    _write_interactive_html(report_path, summary)
    logging.info("Interactive evaluation complete: %s", report_path)


if __name__ == "__main__":
    main()
