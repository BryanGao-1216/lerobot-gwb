#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

args=(
  --policy.type=pi05_lap
  --policy.path=/tmp/gmodels/PI05-base
  --policy.training_stage=pre-training
  --policy.lam_model_path=/gaowenbin/models/base_models/LAM/UniVLA/lam-stage-2.ckpt
  --policy.tokenizer_path=/gaowenbin/models/tokenizers/paligemma-tokenizer
  --policy.push_to_hub=false
  --policy.discrete_state_input=false
  --policy.max_action_dim=32
  --policy.max_state_dim=32
  --policy.dtype=float32
  --policy.train_expert_only=false
  --policy.attention_backend=flash_attention_2
  --policy.freeze_vision_encoder=true
  --policy.gradient_checkpointing=false
  --policy.push_to_hub=false
  --policy.chunk_size=12
  --policy.n_action_steps=12
  --policy.optimizer_lr=1e-5
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_decay_lr=1e-6
  --eval_steps=500
  --steps=100000
  --save_freq=10000
  --batch_size=4
  --policy.gradient_accumulation=8
  --resume=false
  --output_dir=/gaowenbin/models/trained_models/PI05-lap-pretrain
  --dataset.config_path=dataset/config.json
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
  --main_process_port="${MAIN_PROCESS_PORT:-29501}" \
  -m lerobot.scripts.lerobot_train \
  "${args[@]}" "$@"
