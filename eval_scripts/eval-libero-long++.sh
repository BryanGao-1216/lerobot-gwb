
export CUDA_VISIBLE_DEVICES=0


python /gaowenbin/projects/lerobot-gwb/eval_scripts/run_libero_long_plus_plus.py \
  --checkpoint /gaowenbin/models/trained_models/PI05-lap-baseline/checkpoints/030000/pretrained_model \
  --benchmark-env /root/miniconda3/envs/eval-libero-long-plus \
  --libero-root /gaowenbin/simulators/LIBERO \
  --suite libero_long_plus_plus_all \
  --output-dir="./outputs/eval/pi05_baseline/libero-long-plus-plus-all-with-videos" \
  --save-videos \
  --device cuda
