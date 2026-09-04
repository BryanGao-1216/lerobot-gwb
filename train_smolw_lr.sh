export CUDA_VISIBLE_DEVICES=3,5

HORIZON="${HORIZON:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${HORIZON}}"
MEMORY_STRIDE="${MEMORY_STRIDE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"


OUTPUT_DIR="${OUTPUT_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-stage1}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${OUTPUT_DIR}/tensorboard}"


accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --main_process_port=25901 \
  "$(which lerobot-train)" \
  --policy.path="/data1/gaowenbing/WorkSpace/models/smolw-base" \
  --policy.vidtwin_checkpoint_path="/data1/gaowenbing/WorkSpace/models/vidtwin-libero/checkpoint-best.ckpt" \
  --policy.motion_camera_key="observation.images.image" \
  --policy.motion_horizon="${HORIZON}" \
  --policy.memory_stride="${MEMORY_STRIDE}" \
  --policy.chunk_size="${HORIZON}" \
  --policy.training_stage=world_model \
  --policy.train_expert_only=false \
  --policy.use_peft=false \
  --policy.freeze_vision_encoder=true \
  --policy.vidtwin_sample_posterior=false \
  --policy.motion_loss_weight=1.0 \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset_type=lerobot \
  --dataset.repo_id=local \
  --dataset.root="/data1/gaowenbing/WorkSpace/datasets/LIBERO-Lerobot" \
  --dataset.eval_split=0.0 \
  --output_dir="${OUTPUT_DIR}" \
  --steps=100000 \
  --save_freq=10000 \
  --batch_size="${BATCH_SIZE}" \
  --eval_steps=0 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_grad_clip_norm=5 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.tensorboard_enable=true \
  --policy.tensorboard_log_freq=10 \
  --policy.tensorboard_log_dir="${TENSORBOARD_LOG_DIR}" \
  --policy.push_to_hub=false
