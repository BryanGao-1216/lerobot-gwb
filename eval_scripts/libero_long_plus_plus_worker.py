"""Simulator subprocess for run_libero_long_plus_plus.py (LiLo-VLA Python environment)."""

import json
import sys
from importlib.util import find_spec
from multiprocessing.connection import Connection
from pathlib import Path

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CONTROL_FREQ = 20
WARMUP_STEPS = 20


def load_tasks(benchmark, config):
    definitions = {
        (task["bddl_file"], task["name"].endswith("(Reorder 1)")): task
        for task in config["long_horizon_tasks"]
        if task["benchmark"] == "libero_long_plus_plus"
    }
    tasks = []
    for index in range(benchmark.get_num_tasks()):
        task = benchmark.get_task(index)
        definition = definitions[(Path(task.bddl_file).stem, task.name.endswith("_variant"))]
        tasks.append(
            {
                "index": index,
                "benchmark_task_id": definition["task_id"],
                "name": definition["name"],
                "instruction": task.language,
                "skills": definition["skills"],
            }
        )
    return tasks


class OrderedProgress:
    def __init__(self, env, skills, mappings):
        from libero.libero.envs.predicates import eval_predicate_fn, get_predicate_fn

        self.evaluate = eval_predicate_fn
        self.predicates = []
        self.completed = 0
        # PickedUp is a singleton. Initialize every object's height at this trial's reset.
        get_predicate_fn("pickedup").initial_heights.clear()
        objects = env.object_states_dict
        for skill in skills:
            predicates = []
            for name, *args in mappings[skill]["bddl_predicates"]:
                args = [objects[arg if arg in objects else arg.removesuffix("_main")] for arg in args]
                predicates.append((name, args))
            if not predicates:
                raise ValueError(f"No predicates defined for skill {skill!r}")
            self.predicates.append(predicates)
        self.previous = self.status()

    def status(self):
        return [all(self.evaluate(name, *args) for name, args in skill) for skill in self.predicates]

    def update(self):
        current = self.status()
        activated = [after and not before for before, after in zip(self.previous, current, strict=True)]
        self.previous = current
        while self.completed < len(activated) and activated[self.completed]:
            self.completed += 1
        # Record later skills that became true before the next required skill.
        return [index for index in range(self.completed, len(activated)) if activated[index]]


def pack_observation(env, obs):
    return {
        "pixels": {name: obs[name] for name in CAMERAS},
        "robot_state": {
            "eef": {
                "pos": obs["robot0_eef_pos"][None],
                "quat": obs["robot0_eef_quat"][None],
                "mat": env.robots[0].controller.ee_ori_mat[None],
            },
            "gripper": {"qpos": obs["robot0_gripper_qpos"][None]},
        },
    }


def serve(connection, suite, control_mode, libero_root):
    import numpy as np

    # Prefer the checkout passed as the worker's cwd.
    sys.path.insert(0, str(Path.cwd()))
    if libero_root:
        sys.path.insert(0, libero_root)
        package = find_spec("libero.libero")
        expected = Path(libero_root) / "libero/libero/__init__.py"
        if package is None or package.origin is None or Path(package.origin).resolve() != expected.resolve():
            raise ImportError(
                f"The installed LIBERO shadows --libero-root {libero_root}. Check the simulator Python."
            )
    import lilo_vla.benchmark  # noqa: F401  # Registers suites and applies the benchmark patches.
    from libero.libero.benchmark import get_benchmark
    from lilo_vla import CONFIGS_DIR

    config = json.loads((CONFIGS_DIR / "tasks_and_skills.json").read_text())
    benchmark = get_benchmark(suite)()
    tasks = load_tasks(benchmark, config)
    connection.send(tasks)

    env = None
    active_task = None
    settle_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    try:
        while True:
            command, value = connection.recv()
            if command == "reset":
                if active_task != value:
                    if env is not None:
                        env.close()
                        env = None
                    env, _ = benchmark.make_env(value, control_freq=CONTROL_FREQ)
                    active_task = value
                else:
                    env.reset()
                # Match the upstream pipeline's settling steps before scoring any skills.
                env.robots[0].controller.use_delta = True
                for _ in range(WARMUP_STEPS):
                    obs, _, _, _ = env.step(settle_action)
                env.robots[0].controller.use_delta = control_mode == "relative"
                skills = tasks[value]["skills"]
                progress = OrderedProgress(env.env, skills, config["skill_mappings"])
                connection.send(pack_observation(env, obs))
            elif command == "step":
                obs, _, _, _ = env.step(np.asarray(value, dtype=np.float32))
                out_of_order = progress.update()
                bddl_success = bool(env.check_success())
                success = progress.completed == len(skills) and bddl_success
                connection.send(
                    (
                        None if success or out_of_order else pack_observation(env, obs),
                        {
                            "success": success,
                            "bddl_success": bddl_success,
                            "completed_skills": progress.completed,
                            "out_of_order_skills": [skills[index] for index in out_of_order],
                        },
                    )
                )
            else:
                raise ValueError(f"Unknown simulator command {command!r}")
    except EOFError:
        pass
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    fd, suite, control_mode, libero_root = sys.argv[1:]
    with Connection(int(fd)) as connection:
        serve(connection, suite, control_mode, libero_root)
