"""Offline action evaluation using the policy's complete inference path."""

from bisect import bisect_right
from dataclasses import replace
from typing import Any

import torch
from tqdm import tqdm

from lerobot.processor.converters import batch_to_transition
from lerobot.utils.constants import ACTION

METRIC_NAMES = ("action_rmse", "gripper_accuracy", "cumulative_action_error")


def full_horizon_indices(dataset, horizon: int) -> list[int]:
    """Select held-out starting frames without decoding images or padding an episode tail."""
    if horizon <= 0:
        raise ValueError("Action evaluation requires a positive horizon.")
    selected = []
    if hasattr(dataset, "children"):
        offset = 0
        for _, child, index in dataset.children:
            previous = 0
            starts = child.meta.episodes["dataset_from_index"]
            ends = child.meta.episodes["dataset_to_index"]
            for start, end in zip(index._starts, index._cum_lengths, strict=True):
                episode = bisect_right(starts, int(start)) - 1
                available = min(int(end) - previous, int(ends[episode]) - int(start) - horizon + 1)
                selected.extend(range(offset + previous, offset + previous + max(0, available)))
                previous = int(end)
            offset += len(index)
    else:
        episodes = dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes)
        mapping = dataset.absolute_to_relative_idx
        for episode in episodes:
            start = int(dataset.meta.episodes["dataset_from_index"][episode])
            end = int(dataset.meta.episodes["dataset_to_index"][episode])
            selected.extend(mapping[i] if mapping is not None else i for i in range(start, end - horizon + 1))
    return selected


def action_chunk_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    dim_is_pad: torch.Tensor | None = None,
    *,
    continuous_gripper: bool | torch.Tensor = False,
) -> torch.Tensor:
    """Return [RMSE, gripper accuracy, cumulative MAE] per chunk, averaging arms equally.

    Each arm is six incremental action coordinates followed by a gripper.
    Continuous grippers participate in both error metrics and have NaN accuracy
    (not applicable); discrete grippers keep the original pose-only errors.
    Padding is a contiguous suffix, so a mixed 14-D batch may contain 7-D samples.
    """
    if predicted.ndim != 3 or predicted.shape != target.shape or target.shape[1] < 1:
        raise ValueError(
            "Predicted and target actions must have the same nonempty [batch, horizon, dim] shape."
        )
    predicted, target = predicted.float(), target.to(predicted.device, dtype=torch.float32)
    batch_size, _, width = target.shape
    continuous = torch.as_tensor(continuous_gripper, device=target.device, dtype=torch.bool)
    if continuous.ndim == 0:
        continuous = continuous.expand(batch_size)
    if continuous.shape != (batch_size,):
        raise ValueError("continuous_gripper must be a boolean or have shape [batch].")
    padding = (
        torch.zeros(batch_size, width, dtype=torch.bool, device=target.device)
        if dim_is_pad is None
        else dim_is_pad.to(device=target.device, dtype=torch.bool)
    )
    if padding.shape != (batch_size, width):
        raise ValueError("action_dim_is_pad must have shape [batch, action_dim].")
    dimensions = (~padding).sum(-1)
    expected_padding = torch.arange(width, device=target.device)[None, :] >= dimensions[:, None]
    if not torch.equal(padding, expected_padding) or not ((dimensions == 7) | (dimensions == 14)).all():
        raise ValueError(
            "Action chunk evaluation supports 7 or 14 valid dimensions followed by suffix padding."
        )
    result = torch.empty(batch_size, len(METRIC_NAMES), device=target.device)
    for dimension in (7, 14):
        selected = dimensions == dimension
        if not selected.any():
            continue
        pred = predicted[selected, :, :dimension].reshape(-1, target.shape[1], dimension // 7, 7)
        truth = target[selected, :, :dimension].reshape_as(pred)
        if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
            raise ValueError("Action evaluation encountered non-finite valid action coordinates.")
        error = pred - truth
        for is_continuous, coordinates in ((False, 6), (True, 7)):
            # Select the metric dimensions per sample, including in a mixed-source batch.
            rows = continuous[selected] == is_continuous
            action_error = error[rows, ..., :coordinates]
            indices = selected.nonzero().flatten()[rows]
            result[indices, 0] = action_error.square().mean(dim=(1, 3)).sqrt().mean(dim=1)
            result[indices, 2] = action_error.sum(dim=1).abs().mean(dim=(1, 2))
        accuracy = ((pred[..., 6] > 0) == (truth[..., 6] > 0)).float().mean(dim=(1, 2))
        result[selected, 1] = torch.where(continuous[selected], float("nan"), accuracy)
    return result


@torch.no_grad()
def evaluate_action_chunks(
    policy,
    preprocessor,
    postprocessor,
    dataloader,
    accelerator,
    source_names: list[str],
    *,
    seed: int = 0,
    gripper_representations: list[str] | None = None,
) -> dict[str, float]:
    """Run complete inference, unnormalize per source, and gather unpadded sample metrics.

    Call through the prepared policy's forward so DDP/FSDP hooks remain active.
    The action postprocessor receives dataset_index as well as the predicted action;
    calling a tensor-only postprocessor would lose the source normalization selection.
    """
    if gripper_representations is None:
        gripper_representations = ["discrete"] * len(source_names)
    if len(gripper_representations) != len(source_names) or any(
        value not in ("discrete", "continuous") for value in gripper_representations
    ):
        raise ValueError("Provide one 'discrete' or 'continuous' gripper representation per source.")
    config = accelerator.unwrap_model(policy).config
    # Share the real postprocessor steps/hooks, adapting only the input converter.
    action_postprocessor = replace(postprocessor, to_transition=batch_to_transition)
    totals = torch.zeros(len(source_names), len(METRIC_NAMES), dtype=torch.float64)
    metric_counts = torch.zeros_like(totals)
    chunk_counts = torch.zeros(len(source_names), dtype=torch.int64)
    device = accelerator.device
    continuous_sources = torch.tensor(
        [value == "continuous" for value in gripper_representations], device=device, dtype=torch.bool
    )
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    was_training = policy.training
    policy.eval()
    try:
        # Dataloader setup/inference must not consume the training RNG stream.
        with torch.random.fork_rng(devices=cuda_devices):
            for batch in tqdm(
                dataloader,
                desc="Action chunk evaluation",
                leave=False,
                disable=not accelerator.is_main_process,
            ):
                target = batch[ACTION].detach().clone()
                if "action_is_pad" in batch and batch["action_is_pad"].any():
                    raise ValueError(
                        "Action evaluation received a padded horizon; filter episode tails first."
                    )
                source_ids = batch.get(
                    "dataset_index", torch.zeros(len(target), dtype=torch.long, device=device)
                )
                model_batch: dict[str, Any] = {key: value for key, value in batch.items() if key != ACTION}
                for key, value in model_batch.items():
                    if key in config.image_features and value.dtype == torch.uint8:
                        model_batch[key] = value.float().div(255)
                model_batch = preprocessor(model_batch)
                model_batch.pop(ACTION, None)
                # A sample's noise stays fixed across batches, ranks and evaluation steps.
                noise = []
                for index in batch["index"].detach().cpu().tolist():
                    generator = torch.Generator(device=device).manual_seed(
                        (seed + int(index) * 1000003) % (2**63 - 1)
                    )
                    noise.append(
                        torch.randn(
                            config.chunk_size, config.max_action_dim, generator=generator, device=device
                        )
                    )
                with accelerator.autocast():
                    prediction = policy(model_batch, action_inference=True, noise=torch.stack(noise))
                prediction = action_postprocessor({ACTION: prediction, "dataset_index": source_ids})
                metrics = action_chunk_metrics(
                    prediction,
                    target,
                    batch.get("action_dim_is_pad"),
                    continuous_gripper=continuous_sources[source_ids.to(device=device, dtype=torch.long)],
                )
                # Accelerate removes samples duplicated to equalize distributed final batches.
                metrics, source_ids = accelerator.gather_for_metrics(
                    (metrics.to(device), source_ids.to(device))
                )
                metrics, source_ids = metrics.cpu().double(), source_ids.cpu()
                for source_id in source_ids.unique().tolist():
                    rows = source_ids == source_id
                    values = metrics[rows]
                    valid = ~values.isnan()
                    totals[source_id] += torch.where(valid, values, 0.0).sum(0)
                    metric_counts[source_id] += valid.sum(0)
                    chunk_counts[source_id] += rows.sum()
    finally:
        policy.train(was_training)
    output = {}
    if chunk_counts.sum() == 0:
        raise ValueError("No complete action chunks are available for evaluation.")
    summaries = [("", totals.sum(0), metric_counts.sum(0), chunk_counts.sum())]
    # A single source is identical to the overall result. Logging both would
    # create duplicate charts for every metric in SwanLab/W&B.
    if len(source_names) > 1:
        summaries.extend(zip(source_names, totals, metric_counts, chunk_counts, strict=True))
    for name, values, counts, chunks in summaries:
        if chunks == 0:
            continue
        prefix = f"{name}/" if name else ""
        output[f"{prefix}num_action_chunks"] = int(chunks)
        output.update(
            {
                f"{prefix}{key}": float(value / count)
                for key, value, count in zip(METRIC_NAMES, values, counts, strict=True)
                if count > 0
            }
        )
    return output
