#!/bin/sh

set -eu

export CUDA_VISIBLE_DEVICES=4,5

HORIZON="${HORIZON:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${HORIZON}}"
MEMORY_STRIDE="${MEMORY_STRIDE:-1}"
TRAIN_MODE="${TRAIN_MODE:-action_only}"
Z_CONDITION_WARMUP_STEPS="${Z_CONDITION_WARMUP_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

POLICY_PATH="${POLICY_PATH:-/data1/gaowenbing/WorkSpace/models/smolw-base}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-${TRAIN_MODE}}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${OUTPUT_DIR}/tensorboard}"


accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=25901 \
  "$(which lerobot-train)" \
  --policy.path="${POLICY_PATH}" \
  --policy.vidtwin_checkpoint_path="/data1/gaowenbing/WorkSpace/models/vidtwin-libero/checkpoint-best.ckpt" \
  --policy.motion_camera_key="observation.images.image" \
  --policy.motion_horizon="${HORIZON}" \
  --policy.memory_stride="${MEMORY_STRIDE}" \
  --policy.chunk_size="${HORIZON}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.drop_n_last_frames="${HORIZON}" \
  --policy.train_mode="${TRAIN_MODE}" \
  --policy.z_condition_warmup_steps="${Z_CONDITION_WARMUP_STEPS}" \
  --policy.use_peft=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_state_proj=true \
  --policy.vidtwin_sample_posterior=false \
  --policy.motion_loss_weight=1.0 \
  --policy.z_loss_weight=1.0 \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset_type=lerobot \
  --dataset.repo_id=local \
  --dataset.root="/data1/gaowenbing/WorkSpace/datasets/LIBERO-Lerobot" \
  --dataset.eval_split=0.0 \
  --output_dir="${OUTPUT_DIR}" \
  --steps=100000 \
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
