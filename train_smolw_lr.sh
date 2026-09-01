#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# The launch script owns every external location; policy code contains no
# machine-specific model or dataset directory.
SMOLW_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SMOLW_MODEL_DIR="${SMOLW_MODEL_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-base}"
VIDTWIN_REPO_DIR="${VIDTWIN_REPO_DIR:-${SMOLW_SCRIPT_DIR}/../CoWVLA}"
VIDTWIN_CONFIG_PATH="${VIDTWIN_CONFIG_PATH:-${VIDTWIN_REPO_DIR}/vidtwin/configs/vidtwin_structure_7_7_8_dynamics_7_8.yaml}"
VIDTWIN_CHECKPOINT_PATH="${VIDTWIN_CHECKPOINT_PATH:-/mnt/models/microsoft__vidtwin/main/checkpoints/vidtwin_structure_7_7_8_dynamics_7_8.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/data1/gaowenbing/WorkSpace/datasets/LIBERO-Lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-libero_only}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-libero}"

HORIZON="${HORIZON:-20}"
MEMORY_STRIDE="${MEMORY_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

MOTION_CAMERA_ARGS=()
if [[ -n "${MOTION_CAMERA_KEY:-}" ]]; then
  MOTION_CAMERA_ARGS+=(--policy.motion_camera_key="${MOTION_CAMERA_KEY}")
fi

lerobot-train \
  --policy.path="${SMOLW_MODEL_DIR}" \
  --policy.vidtwin_repo_path="${VIDTWIN_REPO_DIR}" \
  --policy.vidtwin_config_path="${VIDTWIN_CONFIG_PATH}" \
  --policy.vidtwin_checkpoint_path="${VIDTWIN_CHECKPOINT_PATH}" \
  --policy.motion_horizon="${HORIZON}" \
  --policy.memory_stride="${MEMORY_STRIDE}" \
  --policy.chunk_size="${HORIZON}" \
  --policy.n_action_steps="${HORIZON}" \
  --policy.drop_n_last_frames="$((HORIZON - 1))" \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset_type=lerobot \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.eval_split=0.0 \
  --output_dir="${OUTPUT_DIR}" \
  --steps=60000 \
  --save_freq=10000 \
  --batch_size="${BATCH_SIZE}" \
  --eval_steps=0 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_grad_clip_norm=5 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.push_to_hub=false \
  "${MOTION_CAMERA_ARGS[@]}"
