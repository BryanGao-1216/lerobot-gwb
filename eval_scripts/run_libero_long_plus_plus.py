#!/usr/bin/env python3
"""Evaluate a LeRobot LIBERO policy with LiLo-VLA's LIBERO-Long++ simulator."""

import argparse
import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from statistics import mean

from libero_long_plus_plus_worker import CAMERAS, CONTROL_FREQ, WARMUP_STEPS

BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "benchmark" / "LiLo-VLA"
ORIGINAL = "libero_long_plus_plus"
SUITES = (f"{ORIGINAL}_all", ORIGINAL, f"{ORIGINAL}_variant")


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="LeRobot pretrained_model directory or Hugging Face repo ID")
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT, help="LiLo-VLA checkout")
    parser.add_argument(
        "--benchmark-python", type=Path, help="simulator interpreter (defaults to current Python)"
    )
    parser.add_argument(
        "--libero-root", type=Path, help="LIBERO f78abd6 source checkout if needed for imports"
    )
    parser.add_argument("--suite", choices=SUITES, default=SUITES[0])
    parser.add_argument(
        "--list-tasks", action="store_true", help="print suite indices without loading a policy"
    )
    parser.add_argument("--task-id", type=int, action="append", help="zero-based suite index, repeatable")
    parser.add_argument("--episodes", type=positive_int, default=10, help="trials per task (default: 10)")
    parser.add_argument("--max-steps", type=positive_int, default=520, help="per trial (default: 520)")
    parser.add_argument("--device", help="policy device, e.g. cuda, mps, cpu")
    parser.add_argument(
        "--n-action-steps", type=positive_int, help="override the policy's action chunk execution length"
    )
    parser.add_argument("--policy-override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--agent-camera", help="checkpoint feature suffix for the scene camera (auto-detected)"
    )
    parser.add_argument(
        "--wrist-camera", help="checkpoint feature suffix for the wrist camera (auto-detected)"
    )
    parser.add_argument("--control-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--output-dir", type=Path, help="new directory for eval_info.json")
    args = parser.parse_args()
    if not args.list_tasks and not args.checkpoint:
        parser.error("--checkpoint is required for evaluation")
    if any("=" not in override for override in args.policy_override):
        parser.error("--policy-override must have the form KEY=VALUE")

    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    if not (args.benchmark_root / "lilo_vla" / "benchmark" / "__init__.py").is_file():
        parser.error(f"LiLo-VLA checkout not found at {args.benchmark_root}")
    args.benchmark_python = args.benchmark_python or Path(sys.executable)
    args.benchmark_python = args.benchmark_python.expanduser().absolute()
    if not args.benchmark_python.is_file():
        parser.error(
            f"Simulator Python not found at {args.benchmark_python}. See README.libero-long-plus-plus.md."
        )
    if args.libero_root is not None:
        args.libero_root = args.libero_root.expanduser().resolve()
        if not (args.libero_root / "libero/libero/__init__.py").is_file():
            parser.error(f"LIBERO source checkout not found at {args.libero_root}")
    return parser, args


@contextmanager
def open_simulator(args):
    worker_env = os.environ.copy()
    worker_env.pop("LIBERO_CONFIG_PATH", None)  # Let LiLo-VLA configure its own assets and BDDL paths.
    if sys.platform == "linux":
        worker_env.setdefault("MUJOCO_GL", "egl")
    parent, child = socket.socketpair()
    with parent, child:
        worker = subprocess.Popen(
            [
                str(args.benchmark_python),
                "-E",
                str(Path(__file__).resolve().with_name("libero_long_plus_plus_worker.py")),
                str(child.fileno()),
                args.suite,
                args.control_mode,
                str(args.libero_root) if args.libero_root is not None else "",
            ],
            cwd=args.benchmark_root,
            env=worker_env,
            pass_fds=(child.fileno(),),
            start_new_session=True,
            stdout=sys.stderr,
        )
        child.close()
        try:
            with Connection(parent.detach()) as connection:
                yield connection
        finally:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


class LeRobotPolicy:
    def __init__(self, args):
        from lerobot.configs import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.utils.constants import ACTION, OBS_IMAGES
        from lerobot.utils.device_utils import is_torch_device_available
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        checkpoint_path = Path(args.checkpoint).expanduser()
        if (
            not (checkpoint_path / "config.json").is_file()
            and (checkpoint_path / "pretrained_model").is_dir()
        ):
            checkpoint_path /= "pretrained_model"
        self.checkpoint = str(checkpoint_path.resolve()) if checkpoint_path.is_dir() else args.checkpoint
        overrides = [f"--{item}" for item in args.policy_override]
        if args.device:
            if not is_torch_device_available(args.device):
                raise ValueError(f"Policy device {args.device!r} is unavailable")
            overrides.append(f"--device={args.device}")
        if args.n_action_steps:
            overrides.append(f"--n_action_steps={args.n_action_steps}")
        config = PreTrainedConfig.from_pretrained(self.checkpoint, cli_overrides=overrides)
        config.pretrained_path = self.checkpoint
        if config.type == "pi05_lap" and config.training_stage == "pre-training":
            raise ValueError("LIBERO evaluation needs a pi05_lap post-training action checkpoint")
        action = (config.output_features or {}).get(ACTION)
        if action is not None and action.shape != (7,) and config.type != "xvla":
            raise ValueError(f"LIBERO expects 7D actions, checkpoint declares {action.shape}")

        def camera_suffix(requested, candidates):
            return requested or next(
                (name for name in candidates if f"{OBS_IMAGES}.{name}" in config.image_features),
                candidates[0],
            )

        self.camera_map = {
            CAMERAS[0]: camera_suffix(args.agent_camera, ("image", "base_0_rgb", CAMERAS[0])),
            CAMERAS[1]: camera_suffix(args.wrist_camera, ("image2", "left_wrist_0_rgb", CAMERAS[1])),
        }
        if len(set(self.camera_map.values())) != len(self.camera_map):
            raise ValueError("Scene and wrist cameras must have different feature suffixes")
        env_cfg = LiberoEnv(
            camera_name_mapping=self.camera_map,
            control_mode=args.control_mode,
            observation_height=256,
            observation_width=256,
        )
        self.policy = make_policy(cfg=config, env_cfg=env_cfg).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(config.device)}},
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(env_cfg, config)

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.env_preprocessor.reset()
        self.env_postprocessor.reset()

    def action(self, observation, instruction):
        import torch

        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        observation["pixels"] = {
            target: observation["pixels"][source] for source, target in self.camera_map.items()
        }
        batch = preprocess_observation(observation)
        batch["task"] = [instruction]
        batch = self.preprocessor(self.env_preprocessor(batch))
        action = self.postprocessor(self.policy.select_action(batch))
        action = self.env_postprocessor({ACTION: action})[ACTION].float().cpu()
        if action.shape != (1, 7) or not torch.isfinite(action).all():
            raise ValueError(f"Expected a finite (1, 7) LIBERO action, got {action}")
        return action[0].tolist()


def summarize(episodes):
    return {
        "episodes": len(episodes),
        "success_rate": mean(episode["success"] for episode in episodes),
        "avg_progress": mean(episode["progress"] for episode in episodes),
        "bddl_success_rate": mean(episode["bddl_success"] for episode in episodes),
    }


def run(args, connection, tasks):
    import torch

    print(f"Loading LeRobot policy {args.checkpoint}", flush=True)
    policy = LeRobotPolicy(args)
    config = policy.policy.config
    output_dir = args.output_dir or Path("outputs/eval/libero_long_plus_plus") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "eval_info.json"
    result = {
        "suite": args.suite,
        "checkpoint": policy.checkpoint,
        "policy_type": config.type,
        "device": str(config.device),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "policy_overrides": args.policy_override,
        "camera_mapping": policy.camera_map,
        "control_mode": args.control_mode,
        "control_freq_hz": CONTROL_FREQ,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": args.max_steps,
        "episodes_per_task": args.episodes,
        "task_ids": [task["index"] for task in tasks],
        "per_task": [],
    }
    all_episodes = []
    amp = torch.autocast(torch.device(config.device).type) if config.use_amp else nullcontext()
    with torch.inference_mode(), amp:
        for task in tasks:
            episodes = []
            task_result = {**task, "episodes": episodes}
            result["per_task"].append(task_result)
            for trial in range(1, args.episodes + 1):
                policy.reset()
                connection.send(("reset", task["index"]))
                observation = connection.recv()
                steps = 0
                for _ in range(args.max_steps):
                    action = policy.action(observation, task["instruction"])
                    connection.send(("step", action))
                    observation, metrics = connection.recv()
                    steps += 1
                    if observation is None:
                        break
                episode = {
                    **metrics,
                    "trial": trial,
                    "steps": steps,
                    "progress": metrics["completed_skills"] / len(task["skills"]),
                }
                episodes.append(episode)
                all_episodes.append(episode)
                task_result["metrics"] = summarize(episodes)
                result["overall"] = summarize(all_episodes)
                temp_path = result_path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(result, indent=2) + "\n")
                temp_path.replace(result_path)
                print(
                    f"[{task['index']}] trial {trial}/{args.episodes} steps={steps} "
                    f"success={metrics['success']} progress={episode['progress']:.0%}",
                    flush=True,
                )

    print(f"SR={result['overall']['success_rate']:.1%} AP={result['overall']['avg_progress']:.1%}")
    print(result_path)


def main():
    parser, args = parse_args()
    try:
        with open_simulator(args) as connection:
            tasks = connection.recv()
            if args.list_tasks:
                for task in tasks:
                    print(f"{task['index']:2d}  {task['name']}")
                return
            if args.task_id is not None:
                if any(index < 0 or index >= len(tasks) for index in args.task_id):
                    parser.error(f"--task-id must be between 0 and {len(tasks) - 1} for {args.suite}")
                tasks = [tasks[index] for index in dict.fromkeys(args.task_id)]
            run(args, connection, tasks)
    except (EOFError, BrokenPipeError, ConnectionResetError) as error:
        raise SystemExit("LiLo-VLA simulator exited unexpectedly. Check its traceback above.") from error


if __name__ == "__main__":
    main()
