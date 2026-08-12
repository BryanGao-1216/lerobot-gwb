#!/usr/bin/env python

"""Measure per-dataset ActionMem q0 usage with the training RLDS/OXE pipeline.

The script deliberately constructs each OXE source through ``ActionMemRLDSDataset``.
Consequently it uses the same storage selection, OXE standardizer, per-source
q01/q99 action normalization, invalid-chunk filtering, and frozen VQ-VAE encoder
as ``lerobot-train``.  Sources are visited separately so their q0 distributions
can be compared instead of being hidden by the weighted mixture sampler.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class DatasetCodeStatistics:
    dataset: str
    source_format: str
    mixture_weight: float
    valid_chunks: int
    sampled_chunks: int
    counts: list[int]
    probabilities: list[float]
    used_codes: int
    perplexity: float
    normalized_entropy: float
    top_code: int
    top_probability: float
    mean_nearest_q0_distance: float
    mean_soft_target_entropy: float
    mean_soft_target_peak_probability: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the q0 code distribution of every OXE subset using exactly the "
            "ActionMem RLDS action preprocessing used by lerobot-train."
        )
    )
    parser.add_argument("--data-root-dir", type=Path, required=True, help="Root containing OXE datasets.")
    parser.add_argument(
        "--data-mix",
        default="action_tokenizer_plus",
        help="Name in OXE_NAMED_MIXTURES. Ignored when --mixture-path is supplied.",
    )
    parser.add_argument(
        "--mixture-path",
        type=Path,
        default=None,
        help="Optional LeRobot RLDS mixture JSON; uses the same schema as --dataset.rlds_mixture_path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional explicit subset names. Each receives mixture weight 1.0.",
    )
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rlds-storage-format",
        choices=("auto", "hybrid", "tfds", "webdataset"),
        default="hybrid",
        help="Matches --dataset.rlds_storage_format. auto and hybrid select tar per source when available.",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or a concrete device such as cuda:1.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--rlds-num-parallel-calls", type=int, default=16)
    parser.add_argument(
        "--rlds-shuffle-buffer-size",
        type=int,
        default=100_000,
        help="TFDS shuffle size. Tar sources are traversed directly because ordering does not change code counts.",
    )
    parser.add_argument(
        "--camera-views",
        nargs="+",
        choices=("primary", "secondary", "wrist"),
        default=("primary", "secondary", "wrist"),
    )
    parser.add_argument(
        "--soft-target-temperature",
        type=float,
        default=1.0,
        help="Temperature used only for reporting the training soft-target entropy and peak probability.",
    )
    parser.add_argument(
        "--no-balance-weights",
        action="store_true",
        help="Aggregate subsets using raw mixture weights instead of weight × valid chunk count.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first incompatible/missing subset. By default errors are recorded and other subsets continue.",
    )
    args = parser.parse_args()
    for name in ("samples_per_dataset", "batch_size", "state_dim", "rlds_num_parallel_calls"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rlds_shuffle_buffer_size <= 0:
        parser.error("--rlds-shuffle-buffer-size must be positive")
    if args.soft_target_temperature <= 0:
        parser.error("--soft-target-temperature must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


def _distribution_metrics(counts: np.ndarray) -> tuple[np.ndarray, int, float, float, int, float]:
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot summarize an empty q0 distribution.")
    probabilities = counts.astype(np.float64) / total
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum())
    perplexity = float(np.exp(entropy))
    normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    top_code = int(probabilities.argmax())
    return (
        probabilities,
        int(nonzero.sum()),
        perplexity,
        normalized_entropy,
        top_code,
        float(probabilities[top_code]),
    )


def _batch_actions(frames: Iterable[Mapping[str, Any]], batch_size: int, action_key: str):
    batch: list[np.ndarray] = []
    for frame in frames:
        batch.append(np.asarray(frame[action_key], dtype=np.float32))
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch.clear()
    if batch:
        yield np.stack(batch)


def _source_frames(dataset: Any):
    """Yield preprocessed frames without decoding images or filling the large tar shuffle buffer."""
    if dataset._webdataset_sources:
        if len(dataset._webdataset_sources) != 1:
            raise RuntimeError("Per-subset analysis expected exactly one WebDataset source.")
        yield from dataset._iter_webdataset_source(dataset._webdataset_sources[0], 0)
        return
    yield from dataset._iter_tfds_frames()


def _analyze_dataset(
    *,
    dataset: Any,
    dataset_name: str,
    mixture_weight: float,
    samples: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
    progress_every: int,
    action_key: str,
) -> DatasetCodeStatistics:
    encoder = dataset.collate_fn.encoder.to(device)
    encoder.eval()
    codebook_size = int(encoder.codebook_size)
    counts = np.zeros(codebook_size, dtype=np.int64)
    sampled = 0
    distance_sum = 0.0
    entropy_sum = 0.0
    peak_sum = 0.0

    frames = _source_frames(dataset)

    def limited_frames():
        for index, frame in enumerate(frames):
            if index >= samples:
                return
            yield frame

    with torch.inference_mode():
        for action_batch in _batch_actions(limited_frames(), batch_size, action_key):
            actions = torch.from_numpy(action_batch).to(device=device, dtype=torch.float32)
            distances = encoder.compute_code_distances(actions).float()
            if distances.ndim != 2 or distances.shape[1] != codebook_size:
                raise ValueError(
                    f"Expected q0 distances [B, {codebook_size}], got {tuple(distances.shape)} for {dataset_name}."
                )
            codes = distances.argmin(dim=-1)
            counts += torch.bincount(codes, minlength=codebook_size).cpu().numpy()
            probabilities = torch.softmax(-distances / temperature, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(
                dim=-1
            )
            distance_sum += float(distances.min(dim=-1).values.sum().item())
            entropy_sum += float(entropy.sum().item())
            peak_sum += float(probabilities.max(dim=-1).values.sum().item())
            sampled += int(actions.shape[0])
            if progress_every and (sampled // progress_every) != ((sampled - actions.shape[0]) // progress_every):
                logging.info("%s: sampled %d/%d chunks", dataset_name, sampled, samples)

    probabilities, used, perplexity, normalized_entropy, top_code, top_probability = _distribution_metrics(counts)
    return DatasetCodeStatistics(
        dataset=dataset_name,
        source_format=dataset.source_formats[dataset.dataset_names[0]],
        mixture_weight=float(mixture_weight),
        valid_chunks=int(dataset.num_frames),
        sampled_chunks=sampled,
        counts=counts.tolist(),
        probabilities=probabilities.tolist(),
        used_codes=used,
        perplexity=perplexity,
        normalized_entropy=normalized_entropy,
        top_code=top_code,
        top_probability=top_probability,
        mean_nearest_q0_distance=distance_sum / sampled,
        mean_soft_target_entropy=entropy_sum / sampled,
        mean_soft_target_peak_probability=peak_sum / sampled,
    )


def _write_csv(path: Path, statistics: Sequence[DatasetCodeStatistics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "dataset",
                "source_format",
                "mixture_weight",
                "valid_chunks",
                "sampled_chunks",
                "code_id",
                "count",
                "probability",
            ),
        )
        writer.writeheader()
        for item in statistics:
            for code_id, (count, probability) in enumerate(zip(item.counts, item.probabilities, strict=True)):
                writer.writerow(
                    {
                        "dataset": item.dataset,
                        "source_format": item.source_format,
                        "mixture_weight": item.mixture_weight,
                        "valid_chunks": item.valid_chunks,
                        "sampled_chunks": item.sampled_chunks,
                        "code_id": code_id,
                        "count": count,
                        "probability": probability,
                    }
                )


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "PNG output requires matplotlib. Install it with `pip install matplotlib` or "
            "`pip install -e '.[matplotlib-dep]'`."
        ) from exc
    return plt


def _plot_distributions(
    output_dir: Path,
    statistics: Sequence[DatasetCodeStatistics],
    aggregate: np.ndarray,
) -> None:
    plt = _load_matplotlib()
    probabilities = np.asarray([item.probabilities for item in statistics], dtype=np.float64)
    dataset_names = [item.dataset for item in statistics]

    figure_height = max(5.0, 0.42 * len(statistics) + 2.0)
    fig, ax = plt.subplots(figsize=(18, figure_height))
    image = ax.imshow(np.log10(probabilities + 1e-6), aspect="auto", cmap="magma", vmin=-6, vmax=0)
    ax.set_xlabel("Q0 code ID")
    ax.set_ylabel("OXE subset")
    ax.set_title("Per-subset Q0 distribution (color = log10(probability + 1e-6))")
    ax.set_yticks(np.arange(len(dataset_names)), labels=dataset_names)
    ax.set_xticks(np.arange(0, probabilities.shape[1], 16))
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("log10 probability")
    fig.tight_layout()
    fig.savefig(output_dir / "q0_distribution_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(np.arange(len(aggregate)), aggregate, width=1.0)
    ax.set_xlim(-0.5, len(aggregate) - 0.5)
    ax.set_xlabel("Q0 code ID")
    ax.set_ylabel("Probability")
    ax.set_title("Training-weighted aggregate Q0 distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "q0_distribution_training_weighted.png", dpi=180)
    plt.close(fig)

    per_dataset_dir = output_dir / "per_dataset"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)
    for item in statistics:
        fig, ax = plt.subplots(figsize=(18, 4.5))
        ax.bar(np.arange(len(item.probabilities)), item.probabilities, width=1.0)
        ax.set_xlim(-0.5, len(item.probabilities) - 0.5)
        ax.set_xlabel("Q0 code ID")
        ax.set_ylabel("Probability")
        ax.set_title(
            f"{item.dataset} | n={item.sampled_chunks:,}, used={item.used_codes}/{len(item.counts)}, "
            f"ppl={item.perplexity:.1f}, top={item.top_code} ({item.top_probability:.3f})"
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(per_dataset_dir / f"{_safe_filename(item.dataset)}.png", dpi=150)
        plt.close(fig)


def _aggregate_distribution(
    statistics: Sequence[DatasetCodeStatistics], *, balance_weights: bool
) -> tuple[np.ndarray, dict[str, float]]:
    raw_weights = np.asarray(
        [item.mixture_weight * (item.valid_chunks if balance_weights else 1.0) for item in statistics],
        dtype=np.float64,
    )
    normalized_weights = raw_weights / raw_weights.sum()
    probabilities = np.asarray([item.probabilities for item in statistics], dtype=np.float64)
    aggregate = (normalized_weights[:, None] * probabilities).sum(axis=0)
    return aggregate, {
        item.dataset: float(weight) for item, weight in zip(statistics, normalized_weights, strict=True)
    }


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from lerobot.configs.default import DatasetConfig
    from lerobot.datasets.rlds_dataset import (
        _ACTION_VQVAE_INPUT,
        ActionMemRLDSDataset,
        _load_mixture_spec,
        _load_rlds_backend,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device={args.device!r} requested CUDA, but PyTorch cannot access CUDA.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.vqvae_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VQ-VAE checkpoint does not exist: {checkpoint}")

    backend = _load_rlds_backend()
    mixture_config = DatasetConfig(
        repo_id=args.data_mix,
        root=str(args.data_root_dir.expanduser().resolve()),
        rlds_data_mix=args.data_mix,
        rlds_mixture_path=str(args.mixture_path.expanduser().resolve()) if args.mixture_path else None,
        rlds_storage_format=args.rlds_storage_format,
    )
    mixture_spec = (
        [(name, 1.0) for name in args.datasets]
        if args.datasets
        else _load_mixture_spec(mixture_config, backend.named_mixtures)
    )

    # Read the checkpoint metadata without constructing a second encoder model.
    from lerobot.policies.actionmem.action_vqvae import _load_checkpoint

    checkpoint_payload = _load_checkpoint(checkpoint)
    checkpoint_config = checkpoint_payload.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError(f"VQ-VAE checkpoint {checkpoint} has no config mapping.")
    action_horizon = int(checkpoint_config["horizon"])
    action_dim = int(checkpoint_config["action_dim"])
    codebook_size = int(checkpoint_config.get("codebook_size", 256))
    del checkpoint_payload
    gc.collect()

    results: list[DatasetCodeStatistics] = []
    errors: dict[str, str] = {}
    device = torch.device(args.device)
    root = args.data_root_dir.expanduser().resolve()
    logging.info(
        "Analyzing %d OXE subsets with horizon=%d, action_dim=%d, q0_size=%d",
        len(mixture_spec),
        action_horizon,
        action_dim,
        codebook_size,
    )

    for position, (name, mixture_weight) in enumerate(mixture_spec, start=1):
        logging.info("[%d/%d] Building OXE subset %s", position, len(mixture_spec), name)
        dataset = None
        try:
            single_config = DatasetConfig(
                repo_id=name,
                root=str(root),
                rlds_data_mix=name,
                rlds_shuffle_buffer_size=args.rlds_shuffle_buffer_size,
                rlds_balance_weights=False,
                rlds_camera_views=tuple(args.camera_views),
                rlds_skip_unlabeled=True,
                rlds_num_parallel_calls=args.rlds_num_parallel_calls,
                rlds_storage_format=args.rlds_storage_format,
                rlds_q0_device=args.device,
                rlds_state_dim=args.state_dim,
            )
            dataset = ActionMemRLDSDataset(
                dataset_config=single_config,
                action_horizon=action_horizon,
                action_dim=action_dim,
                state_dim=args.state_dim,
                action_vqvae_checkpoint_path=checkpoint,
                seed=args.seed,
            )
            if len(dataset.dataset_names) != 1:
                raise ValueError(
                    f"Subset name {name!r} expands to {dataset.dataset_names}; use --datasets with concrete OXE names."
                )
            result = _analyze_dataset(
                dataset=dataset,
                dataset_name=name,
                mixture_weight=mixture_weight,
                samples=args.samples_per_dataset,
                batch_size=args.batch_size,
                temperature=args.soft_target_temperature,
                device=device,
                progress_every=args.progress_every,
                action_key=_ACTION_VQVAE_INPUT,
            )
            if len(result.counts) != codebook_size:
                raise ValueError(
                    f"Checkpoint declares {codebook_size} q0 codes but encoder returned {len(result.counts)}."
                )
            results.append(result)
            logging.info(
                "%s complete: n=%d, used=%d/%d, ppl=%.1f, top=%d (%.3f), soft_H=%.3f, soft_peak=%.4f",
                name,
                result.sampled_chunks,
                result.used_codes,
                codebook_size,
                result.perplexity,
                result.top_code,
                result.top_probability,
                result.mean_soft_target_entropy,
                result.mean_soft_target_peak_probability,
            )
        except Exception as exc:
            logging.exception("Failed to analyze OXE subset %s", name)
            errors[name] = f"{type(exc).__name__}: {exc}"
            if args.fail_fast:
                raise
        finally:
            del dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        error_path = args.output_dir / "errors.json"
        error_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"No OXE subset completed successfully. Errors were written to {error_path}.")

    aggregate, effective_weights = _aggregate_distribution(
        results, balance_weights=not args.no_balance_weights
    )
    summary = {
        "config": {
            "data_root_dir": str(root),
            "data_mix": args.data_mix,
            "mixture_path": str(args.mixture_path.expanduser().resolve()) if args.mixture_path else None,
            "rlds_storage_format": args.rlds_storage_format,
            "vqvae_checkpoint": str(checkpoint),
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "codebook_size": codebook_size,
            "samples_per_dataset": args.samples_per_dataset,
            "soft_target_temperature": args.soft_target_temperature,
            "balance_weights": not args.no_balance_weights,
            "seed": args.seed,
        },
        "effective_training_weights": effective_weights,
        "training_weighted_probabilities": aggregate.tolist(),
        "datasets": [asdict(item) for item in results],
        "errors": errors,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(args.output_dir / "code_distributions.csv", results)
    if errors:
        (args.output_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    _plot_distributions(args.output_dir, results, aggregate)
    logging.info("Wrote q0 distribution results to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
