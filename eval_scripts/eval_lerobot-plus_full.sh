#!/usr/bin/env bash
# Full LIBERO-plus benchmark: 4 suites, 10,030 variants, 1 rollout/variant.
# Run: bash eval_lerobot-plus_full.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MUJOCO_GL=egl
export MAGICK_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBERO_CONFIG_PATH="$HOME/.libero-plus"
export PYTHONPATH="$HOME/workspace/simulators/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_WRAPPER="/home/gaowenbin/workspace/simulators/lerobot-eval-plus"
POLICY_PATH="/home/gaowenbin/workspace/trained_models/PI05_LAQ/pretrained_model"
RUN_DIR="/home/gaowenbin/workspace/projects/lerobot-gwb/outputs/eval/pi05_laq_libero_plus_full"

# 0 = avoid making ~10,030 rollout videos, while retaining eval_info.json.
# 1 = use the already-working wrapper to save rollout videos too (requires much more disk space).
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"

mkdir -p "$RUN_DIR"

# Pin and record the exact evaluation setup for scientific reproducibility.
{
  echo "benchmark=LIBERO-plus"
  echo "protocol=all_variants_1_episode_each"
  echo "suites=libero_spatial,libero_object,libero_goal,libero_10"
  echo "episodes_per_variant=1"
  echo "seed=1000"
  echo "control_mode=relative"
  echo "n_action_steps=12"
  echo "max_parallel_tasks=1"
  echo "eval_batch_size=1"
  echo "save_videos=$SAVE_VIDEOS"
  echo "policy_path=$POLICY_PATH"
  echo "conda_env=$CONDA_PREFIX"
  echo "libero_plus_git=$(git -C "$HOME/workspace/simulators/LIBERO-plus" rev-parse HEAD)"
  echo "lerobot_git=$(git -C "/home/gaowenbin/workspace/projects/lerobot-gwb" rev-parse HEAD)"
} > "$RUN_DIR/protocol.txt"

# Preflight: the installed fork should expose the published 10,030 variants.
python - <<'PY'
import json
from pathlib import Path
p = Path.home() / 'workspace/simulators/LIBERO-plus/libero/libero/benchmark/task_classification.json'
expected = {'libero_spatial': 2402, 'libero_object': 2518, 'libero_goal': 2591, 'libero_10': 2519}
d = json.loads(p.read_text())
for suite, count in expected.items():
    actual = len(d[suite])
    print(f'{suite}: {actual} variants')
    if actual != count:
        raise SystemExit(f'Unexpected variant count in {suite}: {actual} != {count}. Check fork revision.')
print('Total variants:', sum(expected.values()))
PY

# Optional: set SAVE_VIDEOS=1 to use your exact working wrapper.
# Default: import LeRobot normally, but disable only video rendering in this process.
run_eval() {
  if [[ "$SAVE_VIDEOS" == "1" ]]; then
    "$EVAL_WRAPPER" "$@"
  else
    python "$SCRIPT_DIR/run_lerobot_eval_no_video.py" "$@"
  fi
}

for suite in libero_spatial libero_object libero_goal libero_10; do
  OUT_DIR="$RUN_DIR/$suite"
  mkdir -p "$OUT_DIR"
  if [[ -s "$OUT_DIR/eval_info.json" ]]; then
    # Only skip if all expected per-task results have actually been saved.
    if python - "$OUT_DIR/eval_info.json" "$suite" <<'PY'
import json, sys
p, suite = sys.argv[1:]
expected = {'libero_spatial': 2402, 'libero_object': 2518, 'libero_goal': 2591, 'libero_10': 2519}[suite]
d = json.load(open(p))
a = [x for x in d.get('per_task', []) if x.get('task_group') == suite]
if len(a) != expected or any(len(x.get('metrics', {}).get('successes', [])) != 1 for x in a):
    sys.exit(1)
print(f'Validated complete {suite}: {len(a)} tasks')
PY
    then
      echo "[SKIP] $suite already finished"
      continue
    fi
  fi

  echo "==== Evaluating $suite ====" | tee "$OUT_DIR/status.txt"
  date -Is | tee -a "$OUT_DIR/status.txt"
  # Keep previous log from an interrupted attempt, if it exists.
  if [[ -f "$OUT_DIR/run.log" ]]; then
    mv "$OUT_DIR/run.log" "$OUT_DIR/run.previous.$(date +%Y%m%d_%H%M%S).log"
  fi

  # Inference and environment arguments remain identical to your successful smoke test.
  run_eval \
    --output_dir="$OUT_DIR" \
    --policy.path="$POLICY_PATH" \
    --policy.use_peft=false \
    --env.camera_name_mapping='{"agentview_image":"base_0_rgb","robot0_eye_in_hand_image":"left_wrist_0_rgb"}' \
    --env.type=libero_plus \
    --env.task="$suite" \
    --env.control_mode=relative \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --eval.recording=false \
    --policy.n_action_steps=12 \
    --seed=1000 \
    --job_name="pi05_baseline_libero_plus_${suite}" \
    2>&1 | tee "$OUT_DIR/run.log"

  test -s "$OUT_DIR/eval_info.json" || {
    echo "ERROR: $suite did not produce eval_info.json" >&2
    exit 1
  }
  echo "[DONE] $suite" | tee -a "$OUT_DIR/status.txt"
done

python "$SCRIPT_DIR/summarize_libero_plus.py" \
  --run-dir "$RUN_DIR" \
  --classification "$HOME/workspace/simulators/LIBERO-plus/libero/libero/benchmark/task_classification.json"

echo "All results are in: $RUN_DIR"
