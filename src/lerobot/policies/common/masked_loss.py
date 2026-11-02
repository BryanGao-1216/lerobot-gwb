"""Loss reduction for batches with heterogeneous action dimensions."""

import torch


def reduce_action_loss(losses: torch.Tensor, dimension_padding: torch.Tensor, reduction: str):
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported loss reduction: {reduction}")
    valid = ~dimension_padding.to(device=losses.device, dtype=torch.bool)
    if valid.shape != (losses.shape[0], losses.shape[2]) or not valid.any(dim=-1).all():
        raise ValueError(
            "action_dim_is_pad must have shape (batch, action_dim) with valid dimensions per row."
        )
    valid = valid[:, None, :].expand_as(losses)
    masked = losses.masked_fill(~valid, 0)
    per_sample = masked.sum(dim=(1, 2)) / valid.sum(dim=(1, 2))
    metrics = {
        "loss": per_sample.mean().item(),
        "loss_per_dim": (masked.sum(dim=(0, 1)) / valid.sum(dim=(0, 1)).clamp_min(1)).detach().cpu().tolist(),
    }
    return (per_sample if reduction == "none" else per_sample.mean()), metrics
