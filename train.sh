#!/bin/sh

export CUDA_VISIBLE_DEVICES=3,5

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --main_process_port=29502 \
  -m lerobot.scripts.lerobot_train \
  --policy.path=/data1/gaowenbing/WorkSpace/models/actionmem-pretrained/checkpoints/last/pretrained_model \
  --policy.effect_tokenizer_checkpoint_path=/data1/gaowenbing/WorkSpace/MyStudy/effectTokenizer/outputs/effect_vqvae.pt \
  --policy.use_peft=false \
  --policy.train_expert_only=false \
  --policy.training_stage=vlm_only \
  --policy.freeze_vision_encoder=false \
  --policy.dtype=bfloat16 \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --dataset.rlds_target_control_hz=10 \
  --dataset.repo_id=libero_only \
  --policy.action_token_soft_target_temperature=0.1 \
  --dataset_type=rlds \
  --dataset.rlds_storage_format=hybrid \
  --dataset.root=/data1/gaowenbing/WorkSpace/datasets/OpenX \
  --output_dir=/data1/gaowenbing/WorkSpace/models/actionmem-final \
  --steps=100000 \
  --policy.tensorboard_log_freq=10 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_grad_clip_norm=5 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=5e-6 \
  --batch_size=1 \
  --eval_step=0 \
  --policy.gradient_checkpointing=true \
  --policy.tensorboard_enable=true \
  --policy.push_to_hub=false
