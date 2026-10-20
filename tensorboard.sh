#!/bin/sh

# 与训练脚本的 --output_dir 保持一致，也可指向多个训练输出目录的共同父目录。
tensorboard \
  --logdir=/mnt/data27T/media/gwb/trained_model/smolvla-lap/tensorboard \
  --host=127.0.0.1 \
  --port=6006
