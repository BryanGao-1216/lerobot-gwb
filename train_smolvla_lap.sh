#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

# 预训练：修改 LAM 权重路径、单个相机名和数据集路径。
# --policy.lam_type 支持 univla、lapa、mvp_lam（也接受 mvp-lam）；切换时同步修改权重路径。
# 图像尺寸和码本大小自动选择，无需手动填写 lam_num_tokens / lam_codebook_size。
# 后训练：将 --policy.type 替换为 --policy.path=<预训练checkpoint的pretrained_model目录>，
# 将 --policy.training_stage 改为 post-training，并修改 --output_dir。

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=29501 \
  -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla_lap \
  --policy.training_stage=pre-training \
  --policy.lam_type=mvp_lam \
  --policy.lam_model_path=/home/gaowenbin/workspace/base_models/model/MVP_LAM/mvp_lam.bin \
  --policy.lam_camera_key=observation.images.image \
  --policy.lam_frame_gap=12 \
  --policy.lam_batch_size=8 \
  --policy.vlm_model_name=/home/gaowenbin/workspace/base_models/SmolVLM2-500M \
  --policy.device=cuda \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=12 \
  --policy.n_action_steps=12 \
  --dataset.repo_id=libero_only \
  --dataset.root=/home/gaowenbin/workspace/datasets/libero \
  --output_dir=/home/gaowenbin/workspace/trained_models/smolvla-lap-mvplam \
  --steps=100000 \
  --save_freq=10000 \
  --policy.tensorboard_enable=true \
  --policy.tensorboard_log_freq=10 \
  --batch_size=64 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_grad_clip_norm=5 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.push_to_hub=false
