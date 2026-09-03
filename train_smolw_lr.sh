#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,2,5

HORIZON="${HORIZON:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${HORIZON}}"
MEMORY_STRIDE="${MEMORY_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAINING_STAGE="${TRAINING_STAGE:-world_model}"
POLICY_PATH="${POLICY_PATH:-/data1/gaowenbing/WorkSpace/models/smolw-base}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-libero}"
VIDTWIN_CHECKPOINT_PATH="${VIDTWIN_CHECKPOINT_PATH:-/data1/gaowenbing/WorkSpace/models/VidTwin/checkpoints/vidtwin_structure_7_7_8_dynamics_7_8.ckpt}"
MOTION_CAMERA_KEY="${MOTION_CAMERA_KEY:-observation.images.image}"
TENSORBOARD_ENABLE="${TENSORBOARD_ENABLE:-true}"
TENSORBOARD_LOG_FREQ="${TENSORBOARD_LOG_FREQ:-10}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${OUTPUT_DIR}/tensorboard}"
if [[ -z "${DROP_N_LAST_FRAMES:-}" ]]; then
  if [[ "${TRAINING_STAGE}" == "action_expert_only" ]]; then
    DROP_N_LAST_FRAMES=0
  else
    DROP_N_LAST_FRAMES="${HORIZON}"
  fi
fi

accelerate launch \
  --multi_gpu \
  --num_processes=3 \
  --num_machines=1 \
  --main_process_port=25901 \
  "$(which lerobot-train)" \
  --policy.path="${POLICY_PATH}" \
  --policy.vidtwin_checkpoint_path="${VIDTWIN_CHECKPOINT_PATH}" \
  --policy.motion_camera_key="${MOTION_CAMERA_KEY}" \
  --policy.motion_horizon="${HORIZON}" \
  --policy.memory_stride="${MEMORY_STRIDE}" \
  --policy.chunk_size="${HORIZON}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.drop_n_last_frames="${DROP_N_LAST_FRAMES}" \
  --policy.training_stage="${TRAINING_STAGE}" \
  --policy.train_expert_only=false \
  --policy.use_peft=false \
  --policy.freeze_vision_encoder=true \
  --policy.vidtwin_sample_posterior=false \
  --policy.motion_loss_weight=1.0 \
  --policy.future_visual_loss_weight=1.0 \
  --policy.future_visual_cosine_weight=0.1 \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset_type=lerobot \
  --dataset.repo_id=local \
  --dataset.root="/data1/gaowenbing/WorkSpace/datasets/LIBERO-Lerobot" \
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
  --policy.tensorboard_enable="${TENSORBOARD_ENABLE}" \
  --policy.tensorboard_log_freq="${TENSORBOARD_LOG_FREQ}" \
  --policy.tensorboard_log_dir="${TENSORBOARD_LOG_DIR}" \
  --policy.push_to_hub=false
