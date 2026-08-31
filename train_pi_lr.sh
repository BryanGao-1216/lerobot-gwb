#!/bin/sh

export CUDA_VISIBLE_DEVICES=3,4

# 用两张 GPU 全参数微调原始 PI0（视觉编码器冻结）。
accelerate launch \
  --use_fsdp \
  --num_processes=2 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --fsdp_version=1 \
  --fsdp_sharding_strategy=FULL_SHARD \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=PI0Pytorch \
  --fsdp_backward_prefetch=BACKWARD_PRE \
  --fsdp_forward_prefetch=false \
  --fsdp_use_orig_params=true \
  --fsdp_state_dict_type=FULL_STATE_DICT \
  --main_process_port=29502 \
  -m lerobot.scripts.lerobot_train \
  --policy.path=/data1/gaowenbing/WorkSpace/models/pi0-base \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.dtype=bfloat16 \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --dataset_type=lerobot \
  --dataset.repo_id=local \
  --dataset.root=/data1/gaowenbing/WorkSpace/datasets/LIBERO-Lerobot \
  --output_dir=/data1/gaowenbing/WorkSpace/models/pi0-libero-baseline \
  --steps=100000 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_grad_clip_norm=5 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=5e-6 \
  --batch_size=4 \
  --eval_step=0 \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false
