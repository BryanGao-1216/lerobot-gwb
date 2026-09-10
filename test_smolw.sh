#!/bin/sh
set -eu

export CUDA_VISIBLE_DEVICES=2

# Match train_smolw_lr.sh's output directory. The checkpoint's saved config
# selects the policy mode; never reinterpret trained weights via a CLI override.
train_version="${train_version:-A}"
case "${train_version}" in
  A|B|C|D) ;;
  *) echo "train_version must be A, B, C, or D" >&2; exit 1 ;;
esac
CHECKPOINT_STEP="${CHECKPOINT_STEP:-030000}"
POLICY_PATH="${POLICY_PATH:-/data1/gaowenbing/WorkSpace/models/smolw-union-${train_version}/checkpoints/${CHECKPOINT_STEP}/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval/smolw-${train_version}-${CHECKPOINT_STEP}}"

lerobot-eval \
  --output_dir="${OUTPUT_DIR}" \
  --policy.path="${POLICY_PATH}" \
  --policy.use_peft=false \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"image2"}' \
  --env.type=libero \
  --env.task=libero_object,libero_spatial,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --eval.n_episodes=5 \
  --policy.n_action_steps=10
