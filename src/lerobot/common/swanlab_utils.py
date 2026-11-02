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
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.common.wandb_utils import WandBLogger, cfg_to_group

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


class SwanLabLogger:
    """Log offline-training metrics to SwanLab without importing its SDK until enabled."""

    def __init__(self, cfg: "TrainPipelineConfig"):
        try:
            import swanlab
        except ModuleNotFoundError as exc:
            if exc.name != "swanlab":
                raise
            raise ImportError(
                "SwanLab logging requires the training extra: pip install -e '.[training]'."
            ) from exc

        self._swanlab = swanlab
        self.log_dir = Path(cfg.output_dir) / "swanlog"
        self.mode = cfg.swanlab.mode
        self.env_fps = cfg.env.fps if cfg.env else None
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # A W&B checkpoint has no SwanLab ID: start a new experiment at the resumed
        # optimizer step. Never pass its W&B ID to SwanLab.
        run_id = cfg.swanlab.run_id
        self._run = swanlab.init(
            project=cfg.swanlab.project,
            workspace=cfg.swanlab.workspace,
            name=cfg.job_name,
            config=cfg.to_dict(),
            tags=cfg_to_group(cfg, return_list=True),
            log_dir=str(self.log_dir),
            mode=self.mode,
            id=run_id,
            resume="allow" if run_id else None,
        )
        if self.mode != "disabled":
            cfg.swanlab.run_id = self._run.id
        if self.mode == "online":
            logging.info("Track this run on SwanLab --> %s", self._run.url)
        else:
            logging.info("SwanLab mode=%s; local logs: %s", self.mode, self.log_dir)

    def log_dict(self, d: dict, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if self.mode == "disabled":
            return

        data = {}
        for key, value in d.items():
            if isinstance(value, str):
                data[f"{mode}/{key}"] = self._swanlab.Text(value)
            elif isinstance(value, (int, float)):
                data[f"{mode}/{key}"] = value
            else:
                logging.debug("Skipping non-scalar SwanLab metric %s (%s).", key, type(value).__name__)
        if data:
            # Use the completed optimizer step, including after resuming. SwanLab
            # accepts different train/eval metrics logged separately at the same step.
            self._run.log(data, step=step)

    def log_policy(self, checkpoint_dir: Path) -> None:
        # SwanLab's public cloud does not support swanlab.save(). Keep the full
        # LeRobot checkpoint (model, tokenizer, processors, optimizer) on disk.
        logging.info("SwanLab: checkpoint saved locally at %s", checkpoint_dir)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if self.mode == "disabled":
            return

        # SwanLab Video only accepts GIF paths. LeRobot eval produces MP4, so
        # decode with the existing PyAV dependency and retain the original MP4.
        path = Path(video_path)
        if path.suffix.lower() != ".gif":
            import av

            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or self.env_fps or 30)
                frames = [frame.to_image() for frame in container.decode(stream)]
            if not frames:
                logging.warning("Skipping empty eval video: %s", path)
                return
            gif_dir = self.log_dir / "videos"
            gif_dir.mkdir(parents=True, exist_ok=True)
            path = gif_dir / f"{mode}_{step}.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
        self._run.log({f"{mode}/video": self._swanlab.Video(str(path))}, step=step)

    def finish(self) -> None:
        self._run.finish()


def make_experiment_logger(cfg: "TrainPipelineConfig") -> SwanLabLogger | WandBLogger | None:
    """SwanLab takes precedence when resuming a checkpoint with legacy W&B enabled."""
    if cfg.swanlab.enable:
        if cfg.wandb.enable:
            logging.info("SwanLab selected; disabling W&B from the loaded training configuration.")
            cfg.wandb.enable = False
        return SwanLabLogger(cfg)
    if cfg.wandb.enable and cfg.wandb.project:
        return WandBLogger(cfg)
    return None
