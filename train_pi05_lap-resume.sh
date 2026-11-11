#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

accelerate launch --num_processes=1 --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/gaowenbin/models/trained_models/PI05-lap-pretrain/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true \
  --steps=100000 \
  --batch_size=2 \
  --policy.gradient_accumulation=16 \
  --swanlab.run_id j970b44u
