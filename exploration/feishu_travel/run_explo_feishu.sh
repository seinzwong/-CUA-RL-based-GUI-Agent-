#!/usr/bin/env bash
set -e

python exploration/feishu_travel/generate_response_feishu.py \
  --exp_config feishu_travel \
  --agent_config fastchat_explore \
  --split train \
  --part_num 1 \
  --part_idx 0 \
  --iteration_num 2 \
  --save_path exploration/feishu_travel/exploration_outputs/explore
