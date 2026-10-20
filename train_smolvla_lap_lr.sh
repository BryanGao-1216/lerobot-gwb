#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

# 预训练：修改 LAM 权重路径、单个相机名和数据集路径。
# 后训练：将 --policy.type 替换为 --policy.path=<预训练checkpoint的pretrained_model目录>，
# 将 --policy.training_stage 改为 post-training，并修改 --output_dir。

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=29503 \
  -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla_lap \
  --policy.training_stage=pre-training \
  --policy.lam_type=univla \
  --policy.lam_model_path=/mnt/data27T/media/gwb/libero-simulation/仿真环境/model/UniVLA/lam-stage-2.ckpt \
  --policy.lam_camera_key=front \
  --policy.lam_frame_gap=12 \
  --policy.lam_batch_size=8 \
  --policy.vlm_model_name=/mnt/data27T/media/gwb/base_models/smolVLM \
  --policy.device=cuda \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=12 \
  --policy.n_action_steps=12 \
  --dataset_type=lerobot \
  --dataset.repo_id=libero_only \
  --dataset.root=/data1/gaowenbing/WorkSpace/datasets/libero \
  --output_dir=/mnt/data27T/media/gwb/trained_model/smolvla-lap \
  --steps=60000 \
  --save_freq=10000 \
  --batch_size=16 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_grad_clip_norm=5 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=60000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.push_to_hub=false
