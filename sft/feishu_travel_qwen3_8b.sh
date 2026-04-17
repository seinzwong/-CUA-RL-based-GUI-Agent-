#!/usr/bin/env bash
set -euo pipefail

# Feishu CUA SFT entry.
# Ensure sft/data/feishu_travel_sft.json exists first.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-./models/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-ckpt/qwen3_8b_feishu_sft}
DATA_PATH=${DATA_PATH:-data/feishu_travel_sft.json}

python -m fastchat.train.train_lora_llama \
  --model_name_or_path ${MODEL_PATH} \
  --data_path ${DATA_PATH} \
  --bf16 True \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy no \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --flash_attn True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_bias none
