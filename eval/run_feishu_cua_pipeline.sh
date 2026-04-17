#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline for Feishu CUA-Lark mock data flow:
# exploration -> data org -> PRM inference -> RL data org -> (optional) PPO

EXP_CONFIG=${EXP_CONFIG:-feishu_travel}
SPLIT=${SPLIT:-train}
PART_NUM=${PART_NUM:-1}
PART_IDX=${PART_IDX:-0}
ITERATION_NUM=${ITERATION_NUM:-2}

EXPLORE_SAVE_DIR=${EXPLORE_SAVE_DIR:-exploration/feishu_travel/exploration_outputs/explore}
EXPLORE_MERGED=${EXPLORE_MERGED:-exploration/feishu_travel/exploration_outputs/exploration.json}
EXPLORE_TINY=${EXPLORE_TINY:-exploration/feishu_travel/exploration_outputs/exploration_tiny.json}

PRM_BASE_MODEL=${PRM_BASE_MODEL:-ckpt/qwen3_8b_feishu_prm/our_base_model}
PRM_LINEAR_PATH=${PRM_LINEAR_PATH:-ckpt/qwen3_8b_feishu_prm/our_model_state.pt}
PRM_INFER_OUTPUT=${PRM_INFER_OUTPUT:-prm/exploration_inference_results_feishu_travel.json}

RL_OUTPUT=${RL_OUTPUT:-prm/sampled_data_rl_training_feishu_travel.json}
RL_OUTPUT_FLAT=${RL_OUTPUT_FLAT:-prm/sampled_data_rl_training_feishu_travel_flatten.json}
RUN_PPO=${RUN_PPO:-0}

python exploration/feishu_travel/generate_response_feishu.py \
  --exp_config "${EXP_CONFIG}" \
  --split "${SPLIT}" \
  --part_num "${PART_NUM}" \
  --part_idx "${PART_IDX}" \
  --iteration_num "${ITERATION_NUM}" \
  --save_path "${EXPLORE_SAVE_DIR}"

python prm/data_org.py \
  --explore_dir "${EXPLORE_SAVE_DIR}" \
  --output_file "${EXPLORE_MERGED}" \
  --tiny_output_file "${EXPLORE_TINY}"

python prm/inference_prm.py \
  --base_model_path "${PRM_BASE_MODEL}" \
  --linear_path "${PRM_LINEAR_PATH}" \
  --input_data "${EXPLORE_MERGED}" \
  --output_file "${PRM_INFER_OUTPUT}" \
  --chat_model_path "Qwen/Qwen3-8B"

python prm/rl_data_org.py \
  --input_file "${PRM_INFER_OUTPUT}" \
  --output_file "${RL_OUTPUT}" \
  --output_flatten_file "${RL_OUTPUT_FLAT}"

if [[ "${RUN_PPO}" == "1" ]]; then
  bash ppo/train_ppo.sh
fi

printf "Feishu CUA pipeline done. RL dataset: %s\n" "${RL_OUTPUT_FLAT}"
