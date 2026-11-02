#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# 从最后一个 checkpoint 继续预训练；steps 是累计总步数（这里为 50 万）。
# config_path 必须指向 pretrained_model/train_config.json。
# 续训无需 policy.path，LAM、tokenizer 和数据集等配置从 checkpoint 读取。
# 首次切换到 SwanLab 会创建新实验，之后的 checkpoint 会自动保存并复用实验 ID。
# 运行前在当前环境安装依赖并登录：pip install -e '.[training,pi05-lap]'; swanlab login

args=(
  --config_path=/gaowenbin/models/trained_models/PI05-lap-pretrain/checkpoints/last/pretrained_model/train_config.json
  --resume=true
  --steps=500000
  --save_freq=10000
  --batch_size=8
  --dataset.video_backend=pyav
  --wandb.enable=false
  --swanlab.enable=true
  --swanlab.project=pi05-lap
  --swanlab.mode=online
  --job_name=pi05-lap-pretrain
  --log_freq=10
)

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=29501 \
  -m lerobot.scripts.lerobot_train \
  "${args[@]}"
