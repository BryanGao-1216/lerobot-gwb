#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,2,5

HORIZON="${HORIZON:-20}"
MEMORY_STRIDE="${MEMORY_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_MODE="${TRAIN_MODE:-motion_only}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-libero}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${OUTPUT_DIR}/tensorboard}"

case "${TRAIN_MODE}" in
  motion_only|action_only|jointly) ;;
  *)
    echo "TRAIN_MODE must be motion_only, action_only, or jointly; got: ${TRAIN_MODE}" >&2
    exit 2
    ;;
esac

accelerate launch \
  --multi_gpu \
  --num_processes=3 \
  --num_machines=1 \
  --main_process_port=25901 \
  "$(which lerobot-train)" \
  --policy.path="/data1/gaowenbing/WorkSpace/models/smolw-base" \
  --policy.vidtwin_checkpoint_path="/data1/gaowenbing/WorkSpace/models/VidTwin/checkpoints/vidtwin_structure_7_7_8_dynamics_7_8.ckpt" \
  --policy.motion_camera_key="observation.images.image" \
  --policy.motion_horizon="${HORIZON}" \
  --policy.memory_stride="${MEMORY_STRIDE}" \
  --policy.chunk_size="${HORIZON}" \
  --policy.n_action_steps="${HORIZON}" \
  --policy.drop_n_last_frames="${HORIZON}" \
  --policy.train_mode="${TRAIN_MODE}" \
  --policy.z_loss_weight=1.0 \
  --policy.freeze_vision_encoder=true \
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
  --policy.tensorboard_enable=true \
  --policy.tensorboard_log_freq=10 \
  --policy.tensorboard_log_dir="${TENSORBOARD_LOG_DIR}" \
  --policy.push_to_hub=false
