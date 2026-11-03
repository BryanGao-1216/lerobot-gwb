#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

# config_path 必须指向 pretrained_model/train_config.json。
# 续训无需 policy.path，LAM 和 tokenizer 配置从 checkpoint 读取。
# 数据配置选择下方 JSON；新增数据集就在 JSON 的 datasets 中添加一个配置块。
# 不传 dataset.config_path 则沿用 checkpoint 内保存的数据配置快照。
# 首次切换到 SwanLab 会创建新实验，之后的 checkpoint 会自动保存并复用实验 ID。

args=(
  --policy.type=pi05_lap
  --policy.path=/tmp/gmodels/PI05-base
  --policy.training_stage=post-training
  --policy.tokenizer_path=/gaowenbin/models/tokenizers/paligemma-tokenizer
  --policy.push_to_hub=false
  --policy.gradient_checkpointing=false
  --resume=false
  --output_dir=/gaowenbin/models/trained_models/PI05-lap-baseline
  --eval_steps=2000
  --steps=500000
  --save_freq=10000
  --batch_size=4
  --policy.gradient_accumulation=8
  --dataset.config_path=dataset/config.json
  --wandb.enable=false
  --swanlab.enable=true
  --swanlab.project=pi05-lap
  --swanlab.mode=online
  --job_name=pi05-lap-pretrain-baseline
  --log_freq=10
)

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=29501 \
  -m lerobot.scripts.lerobot_train \
  "${args[@]}"
