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

import logging
import math
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig


class TensorBoardLogger:
    """Small SummaryWriter wrapper configured by an ActionMem policy."""

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        *,
        purge_step: int | None = None,
        writer_cls=None,
    ):
        policy_cfg = cfg.policy
        if policy_cfg is None:
            raise ValueError("TensorBoard policy logging requires cfg.policy.")

        configured_log_dir = getattr(policy_cfg, "tensorboard_log_dir", None)
        if configured_log_dir is None:
            log_dir = Path(cfg.output_dir) / "tensorboard"
        else:
            log_dir = Path(configured_log_dir).expanduser()
            if not log_dir.is_absolute():
                log_dir = Path(cfg.output_dir) / log_dir
        self.log_dir = log_dir.resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if writer_cls is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "TensorBoard logging requires the 'tensorboard' package. "
                    "Install the training dependencies with `pip install 'lerobot[training]'`."
                ) from exc
            writer_cls = SummaryWriter

        self.log_freq = int(policy_cfg.tensorboard_log_freq)
        self.histogram_freq = int(policy_cfg.tensorboard_histogram_freq)
        self.log_parameters = bool(policy_cfg.tensorboard_log_parameters)
        self.log_gradients = bool(policy_cfg.tensorboard_log_gradients)
        self.writer = writer_cls(
            log_dir=str(self.log_dir),
            purge_step=purge_step,
            max_queue=int(policy_cfg.tensorboard_max_queue),
            flush_secs=int(policy_cfg.tensorboard_flush_secs),
            filename_suffix=policy_cfg.tensorboard_filename_suffix,
        )
        self.writer.add_text("run/config", f"```\n{pformat(cfg.to_dict())}\n```", global_step=purge_step or 0)
        logging.info(
            "TensorBoard logs will be written to %s",
            colored(str(self.log_dir), "yellow", attrs=["bold"]),
        )

    def should_log_scalars(self, step: int) -> bool:
        return step > 0 and step % self.log_freq == 0

    def should_log_histograms(self, step: int) -> bool:
        return step > 0 and (self.log_parameters or self.log_gradients) and step % self.histogram_freq == 0

    def log_scalars(self, values: dict[str, Any], step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported TensorBoard mode {mode!r}.")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            scalar = float(value)
            if math.isfinite(scalar):
                self.writer.add_scalar(f"{mode}/{name}", scalar, global_step=step)

    def log_model_histograms(self, model: torch.nn.Module, step: int) -> None:
        if not self.should_log_histograms(step):
            return
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if self.log_parameters:
                self._add_histogram(f"parameters/{name}", parameter, step)
            if self.log_gradients and parameter.grad is not None:
                self._add_histogram(f"gradients/{name}", parameter.grad, step)

    def _add_histogram(self, name: str, value: torch.Tensor, step: int) -> None:
        tensor = value.detach().float().cpu()
        if tensor.numel() > 0 and torch.isfinite(tensor).all():
            self.writer.add_histogram(name, tensor, global_step=step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
