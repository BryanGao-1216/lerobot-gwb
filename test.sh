
export CUDA_VISIBLE_DEVICES=1

lerobot-eval \
  --output_dir="./outputs/eval/libero_smol_actionmem_rlds" \
  --env.gripper_action_convention=oxe \
  --policy.path="/data1/gaowenbing/WorkSpace/models/smolvla-libero-baseline-rlds/checkpoints/last/pretrained_model" \
  --policy.use_peft=false \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"image2"}' \
  --env.type=libero \
  --env.task=libero_object,libero_spatial \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.n_action_steps=20
