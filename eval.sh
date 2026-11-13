
export CUDA_VISIBLE_DEVICES=0

lerobot-eval \
  --output_dir="./outputs/eval/pi05_lap_baseline" \
  --policy.path="/home/gaowenbin/workspace/base_models/PI05_baseline/pretrained_model" \
  --policy.use_peft=false \
  --env.camera_name_mapping='{"agentview_image":"base_0_rgb","robot0_eye_in_hand_image":"left_wrist_0_rgb"}' \
  --env.type=libero \
  --env.task=libero_object,libero_spatial,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=2 \
  --eval.batch_size=2 \
  --eval.n_episodes=10 \
  --policy.n_action_steps=12
