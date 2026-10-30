#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

# 预训练：policy.path 填原始 PI05 checkpoint（含 config、模型和 processor 文件）。
# 后训练：policy.path 改为预训练输出的 pretrained_model 目录，
# training_stage 改为 post-training，同时修改 output_dir；无需 LAM 权重。
# H = chunk_size。lam_camera_key 只能填一个；省略则自动选择主视角。
# 使用 PyAV 解码视频，避免 TorchCodec 依赖的系统 FFmpeg 动态库缺失。

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --main_process_port=29501 \
  -m lerobot.scripts.lerobot_train \
  --policy.type=pi05_lap \
  --policy.path=/tmp/g-models/PI05-base \
  --policy.training_stage=pre-training \
  --policy.lam_model_path=/gaowenbin/models/base_models/LAM/UniVLA/lam-stage-2.ckpt \
  --policy.tokenizer_path=/gaowenbin/models/tokenizers/paligemma-tokenizer \
  --policy.lam_camera_key=observation.images.image \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=12 \
  --policy.n_action_steps=12 \
  --dataset.repo_id=libero_only \
  --dataset.root=/gaowenbin/datasets/libero \
  --dataset.video_backend=pyav \
  --output_dir=/gaowenbin/models/trained_models/PI05-lap-pretrain \
  --steps=100000 \
  --save_freq=10000 \
  --batch_size=16 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=pi05-lap \
  --wandb.disable_artifact=true \
  --job_name=pi05-lap-pretrain \
  --log_freq=10 \
