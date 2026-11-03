#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 从原生 PI05-base 开始的动作微调 baseline（新训练，不恢复优化器）。
# 与 train_pi05_lap_s2.sh 使用相同的 LIBERO 输入输出、训练参数和数据划分。
# policy.tokenizer_path 只用于 latent token ID；文本输入使用官方 SentencePiece 模型。
# 动作步数由入口参数/配置决定，例如追加 --policy.chunk_size=12 --policy.n_action_steps=12。

args=(
  --policy.type=pi05_lap
  --policy.path=/tmp/gmodels/PI05-base
  --policy.training_stage=post-training
  --policy.tokenizer_path=/gaowenbin/models/tokenizers/paligemma-tokenizer
  --policy.push_to_hub=false
  --policy.discrete_state_input=false
  --policy.image_augmentation=true
  --policy.use_relative_actions=false
  --policy.max_action_dim=32
  --policy.max_state_dim=32
  --policy.dtype=float32
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.attention_backend=flash_attention_2
  --policy.freeze_vision_encoder=true
  --policy.gradient_checkpointing=false
  --policy.push_to_hub=false
  --policy.chunk_size=12
  --policy.n_action_steps=12
  --policy.optimizer_lr=2.5e-5
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_decay_lr=2.5e-6
  --resume=false
  --output_dir=/gaowenbin/models/trained_models/PI05-lap-baseline-openpi
  --eval_steps=2000
  --steps=30000
  --save_freq=10000
  --batch_size=4
  --policy.gradient_accumulation=8
  --resume=false
  --output_dir=/gaowenbin/models/trained_models/PI05-lap-baseline
  --dataset.config_path=dataset/config.json
  --wandb.enable=false
  --swanlab.enable=true
  --swanlab.project=pi05-lap
  --swanlab.mode=online
  --job_name=pi05-lap-baseline-openpi
  --log_freq=10
)

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port="${MAIN_PROCESS_PORT:-29501}" \
  -m lerobot.scripts.lerobot_train \
  "${args[@]}" "$@"
