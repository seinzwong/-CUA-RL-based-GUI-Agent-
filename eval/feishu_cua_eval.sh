#!/usr/bin/env bash
set -euo pipefail

# Feishu CUA-Lark evaluation script.

TASK=${TASK:-feishu_travel}
SAVE_PATH=${SAVE_PATH:-eval/feishu_cua_eval}
LOGS_PATH=${LOGS_PATH:-${SAVE_PATH}/logs}
MODEL_PATH=${MODEL_PATH:-ckpt/qwen3_8b_feishu_rl_loramerged}
MODEL_NAME=${MODEL_NAME:-qwen3_8b_feishu_rl_loramerged}
CUDA_DEVICE=${CUDA_DEVICE:-0}
FS_WORKER_PORT=${FS_WORKER_PORT:-21012}
WAIT_SECONDS=${WAIT_SECONDS:-45}

mkdir -p "${LOGS_PATH}"

cleanup() {
  if [[ -n "${fs_worker_pid:-}" ]]; then
    kill -9 "${fs_worker_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${fs_controller_pid:-}" ]]; then
    kill -9 "${fs_controller_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Start FastChat controller.
python -u -m fastchat.serve.controller --host 0.0.0.0 >> "${LOGS_PATH}/controller.log" 2>&1 &
fs_controller_pid=$!

# Start model worker.
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -u -m fastchat.serve.vllm_worker \
  --model-path "${MODEL_PATH}" \
  --port "${FS_WORKER_PORT}" \
  --host 0.0.0.0 \
  --worker-address "http://localhost:${FS_WORKER_PORT}" >> "${LOGS_PATH}/model_worker.log" 2>&1 &
fs_worker_pid=$!

sleep "${WAIT_SECONDS}"

# Run evaluation.
python -m eval_agent.main \
  --agent_config fastchat \
  --model_name "${MODEL_NAME}" \
  --exp_config "${TASK}" \
  --split test \
  --override \
  --output_path "${SAVE_PATH}"

printf "Feishu CUA evaluation completed. Output: %s\n" "${SAVE_PATH}"
