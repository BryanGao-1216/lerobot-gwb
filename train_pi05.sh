#!/usr/bin/env bash

EFFECT_TOKENIZER_CHECKPOINT="${EFFECT_TOKENIZER_CHECKPOINT:-/mnt/data27T/media/gwb/MyStudy/effectTokenizer/outputs/effect_vqvae.pt}"

lerobot-train \
  --policy.path=/media/fzx/f2f907fa-be7e-46fd-a2f6-720114ae5359/media/gwb/models/pi05_actionmem-base \
  --policy.effect_tokenizer_checkpoint_path="${EFFECT_TOKENIZER_CHECKPOINT}" \
  --policy.train_expert_only=false \
  --policy.training_stage=vlm_only \
  --peft.method_type=LORA \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --dataset.repo_id=libero_test_0805 \
  --dataset_type=rlds \
  --dataset.rlds_target_control_hz=10 \
  --dataset.root=/media/fzx/f2f907fa-be7e-46fd-a2f6-720114ae5359/media/gwb/datasets/Libero \
  --output_dir=/media/fzx/f2f907fa-be7e-46fd-a2f6-720114ae5359/media/gwb/models/pi05-50000 \
  --steps=50000 \
  --policy.optimizer_grad_clip_norm=5 \
  --batch_size=4 \
  --eval_step=0 \
  --policy.gradient_checkpointing=true \
  --policy.tensorboard_enable=true \
  --policy.push_to_hub=false
