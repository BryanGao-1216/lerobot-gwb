#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

python action_token_eval/evaluate_action_tokens.py \
  --policy-path=/data1/gaowenbing/WorkSpace/models/smol_actionmem-final/checkpoints/last/pretrained_model \
  --effect-tokenizer-checkpoint=/data1/gaowenbing/WorkSpace/MyStudy/effectTokenizer/outputs/effect_vqvae.pt \
  --dataset-root=/data1/gaowenbing/WorkSpace/datasets/OpenX \
  --dataset-repo-id=libero_only \
  --rlds-storage-format=hybrid \
  --target-control-hz=10 \
  --shuffle-buffer-size=4096 \
  --num-parallel-calls=8 \
  --num-samples=10 \
  --top-k=5 \
  --device=cuda \
  --output-dir=/data1/gaowenbing/WorkSpace/models/smol_actionmem-final/action_token_eval
