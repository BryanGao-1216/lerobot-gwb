#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/gaowenbin/models/trained_models/PI05-lap-baseline/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=50000
