export CUDA_VISIBLE_DEVICES=3,4

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --num_machines=1 \
  --main_process_port=29503 \
  "$(which lerobot-train)" \
  --policy.path=/data1/gaowenbing/WorkSpace/models/smolvla \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --dataset_type=rlds \
  --dataset.repo_id=libero_only \
  --dataset.root=/data1/gaowenbing/WorkSpace/datasets/OpenX \
  --dataset.rlds_storage_format=hybrid \
  --dataset.rlds_target_control_hz=20 \
  --output_dir=/data1/gaowenbing/WorkSpace/models/smolvla-libero-baseline-rlds \
  --steps=60000 \
  --save_freq=10000 \
  --batch_size=100 \
  --eval_step=0 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_grad_clip_norm=5 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.push_to_hub=false