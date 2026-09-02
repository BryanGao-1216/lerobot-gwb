
export CUDA_VISIBLE_DEVICES=1

lerobot-eval \
  --output_dir="./outputs/eval/smolw_2w" \
  --policy.path="/data1/gaowenbing/WorkSpace/models/smolw-libero/checkpoints/last/pretrained_model" \
  --policy.use_peft=false \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"image2"}' \
  --env.type=libero \
  --env.task=libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=5 \
  --policy.n_action_steps=20
