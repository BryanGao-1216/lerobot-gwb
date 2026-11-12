#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/gaowenbin/models/trained_models/PI05-lap-posttraining-20klaq/checkpoints/010000/pretrained_model/train_config.json \
  --resume=true \
  --steps=30000 \
  --batch_size=4 \
  --policy.gradient_accumulation=8 \
  --swanlab.run_id ncgjc7zs
