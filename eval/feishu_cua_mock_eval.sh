#!/usr/bin/env bash
set -e

# Minimal mock evaluation for CUA-Lark migration path.
python -m eval_agent.main \
  --exp_config feishu_travel \
  --agent_config fastchat \
  --split test
