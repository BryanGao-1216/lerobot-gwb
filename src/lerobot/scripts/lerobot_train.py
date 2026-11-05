#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + SwanLab)
"""

import dataclasses
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.action_chunk_eval import evaluate_action_chunks, full_horizon_indices
from lerobot.common.swanlab_utils import experiment_logging
from lerobot.common.train_utils import (
    gather_fsdp_state_dicts,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_fsdp_optimizer_state,
    load_training_batch_size,
    load_training_data_batches_seen,
    load_training_num_processes,
    load_training_state,
    push_checkpoint_to_hub,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all


def _metrics_to_python(values: list[Any]) -> list[Any]:
    """Read floating-point metric tensors together after the optimizer update.

    Only detached scalar/vector statistics reach this boundary, never model
    activations. Packing them avoids a separate CUDA synchronization per metric.
    Non-tensor outputs retain their existing types.
    """
    tensors = [value for value in values if isinstance(value, torch.Tensor) and value.is_floating_point()]
    if not tensors:
        return values
    device = tensors[0].device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    packed = torch.cat([value.detach().to(device=device, dtype=dtype).reshape(-1) for value in tensors]).cpu()
    result, offset = [], 0
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            size = value.numel()
            result.append(packed[offset : offset + size].reshape(value.shape).to(dtype=value.dtype).tolist())
            offset += size
        else:
            result.append(value)
    return result


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """Perform one optimizer update from a batch or a lazy iterator of N micro-batches.

    N is accelerator.gradient_accumulation_steps. Only the final micro-batch
    synchronizes, clips and updates weights. Logged losses are sample-weighted
    window means, including when the dataloader ends with a smaller batch.
    Detached statistics stay on-device until this optimizer update completes.
    """
    from accelerate.utils import find_batch_size

    accumulation = accelerator.gradient_accumulation_steps
    if accumulation > 1 and not isinstance(batch, Iterator):
        raise ValueError("Gradient accumulation requires a lazy iterator of micro-batches.")
    batches = batch if isinstance(batch, Iterator) else iter([batch])
    update_s, loss_sum, sample_count = 0.0, None, 0
    scalar_sums, output_dict = {}, {}
    reference_batch_size = None
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for micro_step in range(accumulation):
        try:
            micro_batch = next(batches)
        except StopIteration as exc:
            raise ValueError(f"Expected {accumulation} micro-batches for one optimizer update.") from exc
        # Exclude lazy data loading/preprocessing from the optimizer timing.
        start_time = time.perf_counter()
        tensor_batch = (
            [value for value in micro_batch.values() if isinstance(value, torch.Tensor) and value.ndim > 0]
            if isinstance(micro_batch, dict)
            else micro_batch
        )
        batch_size = find_batch_size(tensor_batch)
        reference_batch_size = reference_batch_size or batch_size
        sample_count += batch_size
        with accelerator.accumulate(policy):
            sample_weights, weight_stats = None, {}
            if sample_weighter is not None:
                sample_weights, weight_stats = sample_weighter.compute_batch_weights(micro_batch)
            with accelerator.autocast():
                if sample_weights is not None:
                    per_sample_loss, micro_output = policy(micro_batch, reduction="none")
                    loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-6)
                else:
                    loss, micro_output = policy(micro_batch)
            micro_output = dict(micro_output or {})
            micro_output.update({f"sample_weight_{key}": value for key, value in weight_stats.items()})
            # backward() divides by N. Correct unequal batch sizes before clipping.
            accelerator.backward(loss * (batch_size / reference_batch_size))
            if accelerator.sync_gradients:
                if micro_step != accumulation - 1:
                    raise RuntimeError(
                        "Gradient synchronization occurred before the accumulation window ended."
                    )
                correction = accumulation * reference_batch_size / sample_count
                if correction != 1:
                    for parameter in policy.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = accelerator.clip_grad_norm_(
                    policy.parameters(), grad_clip_norm if grad_clip_norm > 0 else float("inf")
                )
                with lock if lock is not None else nullcontext():
                    optimizer.step()
                optimizer.zero_grad()
                if not accelerator.optimizer_step_was_skipped:
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
                    if has_method(unwrapped, "update"):
                        unwrapped.update()
            elif micro_step == accumulation - 1:
                raise RuntimeError("The final micro-batch did not synchronize gradients.")
        # Match the old Python-float sum without reading each micro-loss on the CPU.
        metric_dtype = torch.float32 if loss.device.type == "mps" else torch.float64
        detached_loss = loss.detach().to(dtype=metric_dtype)
        if loss_sum is None:
            loss_sum = torch.zeros_like(detached_loss)
        loss_sum.add_(detached_loss, alpha=batch_size)
        for key, value in micro_output.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if value.ndim == 0 and value.is_floating_point():
                    value = value.to(device=loss.device, dtype=metric_dtype)
                    scalar_sums[key] = scalar_sums.get(key, 0.0) + value * batch_size
            elif not isinstance(value, bool) and isinstance(value, (int, float)):
                scalar_sums[key] = scalar_sums.get(key, 0.0) + value * batch_size
            # Non-scalars keep the previous last-micro-batch behavior.
            output_dict[key] = value
        update_s += time.perf_counter() - start_time

    output_dict.update({key: value / sample_count for key, value in scalar_sums.items()})
    # One host read for loss, gradient norm and policy metrics, outside the micro-loop.
    start_time = time.perf_counter()
    loss_value, grad_norm_value, *output_values = _metrics_to_python(
        [loss_sum / sample_count, grad_norm, *output_dict.values()]
    )
    output_dict = dict(zip(output_dict, output_values, strict=True))
    update_s += time.perf_counter() - start_time
    train_metrics.loss = loss_value
    train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = update_s
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    # Aggregate the policy's scalar outputs for logging and rank-reduction across the log window.
    if output_dict:
        train_metrics.update_metrics(output_dict)
    return train_metrics, output_dict


def _training_batches(iterator, count: int, preprocessor, camera_keys: list[str]):
    """Load and process micro-batches lazily, releasing each graph before loading the next."""
    for _ in range(count):
        batch = next(iterator)
        for key in camera_keys:
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float().div(255)
        yield preprocessor(batch)


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    cfg.validate()
    gradient_accumulation = getattr(cfg.trainable_config, "gradient_accumulation", 1)

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        # A low-precision policy dtype explicitly enables the matching autocast mode. For a
        # float32 policy, leave this unset so ``accelerate launch --mixed_precision=...`` remains
        # authoritative. This is required by FSDP1 for mixed-storage models such as PI0: build and
        # flatten uniform FP32 parameters first, then let FSDP use BF16 for computation.
        policy_dtype = getattr(cfg.trainable_config, "dtype", None)
        mixed_precision = {"bfloat16": "bf16", "float16": "fp16"}.get(policy_dtype)
        accelerator = Accelerator(
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=gradient_accumulation, sync_with_dataloader=False
            ),
            step_scheduler_with_optimizer=False,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )
    else:
        # The policy CLI is authoritative, including for programmatic callers.
        accelerator.gradient_accumulation_steps = gradient_accumulation
        accelerator.gradient_state.plugin_kwargs["sync_with_dataloader"] = False
        accelerator.step_scheduler_with_optimizer = False
        accelerator.step = 0

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Keep shutdown handling active during model/dataset initialization as well
    # as training: failures and Ctrl+C must close the experiment too.
    with experiment_logging(cfg, enabled=is_main_process) as experiment_logger:
        _train_with_logger(cfg, accelerator, experiment_logger)


def _train_with_logger(cfg: TrainPipelineConfig, accelerator: "Accelerator", experiment_logger):
    from accelerate.utils import DistributedType

    gradient_accumulation = accelerator.gradient_accumulation_steps
    is_main_process = accelerator.is_main_process
    if experiment_logger is None and is_main_process:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: the global main process downloads once to the shared
    # dataset root, then a barrier lets every other rank read the already-populated copy.
    # LeRobotDataset skips its snapshot_download when try_load() succeeds, so no rank re-downloads.
    if is_main_process:
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    accelerator.wait_for_everyone()

    # Other ranks read from the shared copy populated by the main process.
    if not is_main_process:
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.env_eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # completed optimizer updates, independent of micro-batch count
    data_batches_seen = 0

    if cfg.resume:
        # Under FSDP the optimizer state is sharded and must be loaded after `accelerator.prepare()`
        # (see load_fsdp_optimizer_state below), so skip the optimizer here and load it then.
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler, load_optimizer=not is_fsdp
        )
        saved_data_batches = load_training_data_batches_seen(cfg.checkpoint_path)
        # Before accumulation was supported, each optimizer step consumed one batch.
        data_batches_seen = (
            saved_data_batches if saved_data_batches is not None else step - cfg.data_sampling_start_step
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env,
                policy_cfg=cfg.policy,
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes * gradient_accumulation
        logging.info(
            "Effective batch size: %d per device x %d processes x %d accumulation = %d",
            cfg.batch_size,
            num_processes,
            gradient_accumulation,
            effective_bs,
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    sampler_state = None
    if not cfg.dataset.streaming:
        # Native datasets use episode-aware shuffling; configured mixtures use weighted sampling.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across ranks via BatchSamplerShard
        # without needing a `generator` attribute to synchronize an RNG, and resume is sample-exact.
        shuffle = False
        if cfg.dataset.datasets:
            sampler = dataset.make_sampler(
                seed=cfg.seed if cfg.seed is not None else 0,
                batch_multiple=cfg.batch_size * accelerator.num_processes,
            )
        else:
            sampler = EpisodeAwareSampler(
                dataset.meta.episodes["dataset_from_index"],
                dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=dataset.episodes,
                drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
                shuffle=True,
                seed=cfg.seed if cfg.seed is not None else 0,
                absolute_to_relative_idx=dataset.absolute_to_relative_idx,
            )
        if getattr(cfg, "_reset_data_order", False):
            cfg.data_sampling_start_step = step
            data_batches_seen = 0
            if is_main_process:
                logging.warning(
                    "Dataset mixture changed on resume: keeping training state, restarting data sampling at epoch 0."
                )
        if cfg.resume and step > 0 and not getattr(cfg, "_reset_data_order", False):
            # The resume offset depends on the (num_processes, batch_size) that produced `step`, so
            # use the values recorded in the checkpoint (falling back to the current ones for older
            # ckpts that did not store them).
            saved_num_processes = load_training_num_processes(cfg.checkpoint_path)
            saved_batch_size = load_training_batch_size(cfg.checkpoint_path)
            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process and saved_num_processes not in (None, accelerator.num_processes):
                logging.warning(
                    f"Resuming with num_processes={accelerator.num_processes} but the checkpoint was "
                    f"written with num_processes={saved_num_processes}. The data order resumes at the "
                    "right epoch/offset, but per-rank sample-exactness requires the same world size."
                )
            if is_main_process and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written with "
                    f"batch_size={saved_batch_size}. The data order resumes at the right epoch/offset, "
                    "but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(
                data_batches_seen, len(sampler), ckpt_batch_size, ckpt_num_processes
            )
            sampler.load_state_dict(sampler_state)
            if is_main_process:
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = True
        sampler = None

    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    action_chunk_eval = (
        cfg.policy is not None
        and cfg.policy.type in {"pi05", "pi05_lap"}
        and getattr(cfg.policy, "training_stage", "post-training") == "post-training"
    )
    # Build eval dataloader if a held-out split exists.
    eval_dataloader = None
    if cfg.eval_steps > 0 and eval_dataset is None:
        raise ValueError("eval_steps > 0 requires a nonempty held-out dataset from eval_split.")
    if eval_dataset is not None and cfg.eval_steps > 0:
        eval_ds = eval_dataset
        if action_chunk_eval:
            dimensions = (
                {
                    name: sum(len(indices) for _, indices in eval_dataset.schema.action_fields[name])
                    for name, _, _ in eval_dataset.children
                }
                if cfg.dataset.datasets
                else {cfg.dataset.repo_id: eval_dataset.meta.features["action"]["shape"][0]}
            )
            if any(dimension not in (7, 14) for dimension in dimensions.values()):
                raise ValueError(
                    f"Action chunk evaluation requires 7 or 14 mapped action dimensions: {dimensions}."
                )
            indices = full_horizon_indices(eval_dataset, cfg.policy.chunk_size)
            if not indices:
                raise ValueError(
                    "No held-out episode has a complete action horizon; adjust eval_split or chunk_size."
                )
            eval_ds = torch.utils.data.Subset(eval_dataset, indices)
            logging.info(
                "Full action evaluation: %d complete held-out chunks (H=%d).",
                len(indices),
                cfg.policy.chunk_size,
            )
        elif cfg.max_eval_samples > 0 and cfg.dataset.datasets:
            eval_ds = torch.utils.data.Subset(eval_dataset, eval_dataset.eval_indices(cfg.max_eval_samples))
        elif cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            generator=torch.Generator().manual_seed(cfg.seed or 0),
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )

    # FSDP optimizer state is sharded across ranks, so it can only be loaded once the optimizer and
    # model are FSDP-wrapped (i.e. after `prepare`). Collective: every rank must participate.
    if cfg.resume and accelerator.distributed_type == DistributedType.FSDP:
        load_fsdp_optimizer_state(policy, optimizer, cfg.checkpoint_path)

    if sampler_state is not None:
        # Accelerate calls sampler.set_epoch(self.iteration) when iteration begins.
        # Restore its epoch as well, so it does not overwrite the resumed sampler with epoch 0.
        dataloader.set_epoch(sampler_state["epoch"])
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss DDP
        # is actually optimizing. grad_norm and lr are already identical on every rank (post
        # gradient sync / deterministic scheduler) so reducing them would be a no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        # Derived from the post-reduce max step time; set once per log window on the main rank.
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes * gradient_accumulation
    train_tracker = MetricsTracker(
        cfg.batch_size * gradient_accumulation,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )
    # Changing N on resume must not multiply all historical samples by the new N.
    train_tracker.samples = data_batches_seen * cfg.batch_size * accelerator.num_processes
    train_tracker.episodes = train_tracker.samples / (dataset.num_frames / dataset.num_episodes)
    train_tracker.epochs = train_tracker.samples / dataset.num_frames

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    while step < cfg.steps:
        start_time = time.perf_counter()
        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            _training_batches(dl_iter, gradient_accumulation, preprocessor, dataset.meta.camera_keys),
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )
        data_batches_seen += gradient_accumulation
        train_tracker.dataloading_s = max(0.0, time.perf_counter() - start_time - train_tracker.update_s.val)
        if accelerator.optimizer_step_was_skipped:
            # A scaler overflow consumes data, but does not advance schedules/eval/checkpoints.
            continue

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process:
                # Cluster-wide throughput, derived from the already-reduced (max) step time so it
                # reflects the slowest rank — which is what actually gates the next iteration.
                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    train_tracker.samples_per_s = effective_batch_size / step_time
                logging.info(train_tracker)
                if experiment_logger:
                    # Policy sub-losses (latent_loss, action_loss, ...) are aggregated into the
                    # tracker by update_policy, so to_dict() already carries their windowed,
                    # rank-reduced averages — no per-step output_dict passthrough needed.
                    log_dict = train_tracker.to_dict()
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    experiment_logger.log_dict(log_dict, step)
            train_tracker.reset_averages()

        if is_eval_step and action_chunk_eval:
            source_names = list(eval_dataset.sources) if cfg.dataset.datasets else [cfg.dataset.repo_id]
            gripper_representations = (
                [cfg.dataset.datasets[name].gripper_type for name in source_names]
                if cfg.dataset.datasets
                else None
            )
            eval_metrics = evaluate_action_chunks(
                policy,
                preprocessor,
                postprocessor,
                eval_dataloader,
                accelerator,
                source_names,
                seed=cfg.seed or 0,
                gripper_representations=gripper_representations,
            )
            if is_main_process:
                logging.info("step %d: action evaluation %s", step, eval_metrics)
                if experiment_logger:
                    experiment_logger.log_dict(eval_metrics, step=step, mode="eval")

        elif is_eval_step:
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy.forward(eval_batch)
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process:
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if experiment_logger:
                    experiment_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            # Under FSDP, gathering the full model + optimizer state dicts is a cross-rank collective,
            # so all ranks must participate; rank 0 then writes the materialized dicts. For DDP /
            # single-GPU the state dicts are saved the normal way inside save_checkpoint.
            is_fsdp = accelerator.distributed_type == DistributedType.FSDP
            if is_fsdp:
                model_state_dict, optim_state_dict = gather_fsdp_state_dicts(policy, optimizer)
            else:
                model_state_dict, optim_state_dict = None, None
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                    data_batches_seen=data_batches_seen,
                )
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if experiment_logger:
                    experiment_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if experiment_logger:
                    log_dict = {**eval_tracker.to_dict(), **eval_info}
                    experiment_logger.log_dict(log_dict, step, mode="eval")
                    experiment_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    model_state_dict = accelerator.get_state_dict(policy) if is_fsdp else None
    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model, dataset_meta=dataset.meta)
            else:
                unwrapped_model.push_model_to_hub(cfg, state_dict=model_state_dict, dataset_meta=dataset.meta)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """True when the CLI requests a remote HF Jobs run (--job.target=<non-local>)."""
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
