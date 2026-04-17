# PPO training entry for Feishu CUA-Lark pipeline.

export PYTHONPATH=./
export TRAIN_PATH="data_train"
export TRAIN_SET="step_grained_for_ppo_example"
export CUDA_VISIBLE_DEVICES="0"

export MODEL_TYPE="qwen3-8b-feishu"
export MODEL_PATH="./ckpt/qwen3_8b_feishu_sft_loramerged"
export PPO_DATA_FILE="prm/sampled_data_rl_training_feishu_travel_flatten.json"


python ppo/step_ppo.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --config_path config/StepTool_ppo.json \
    --data_file ${PPO_DATA_FILE} \
    --epochs 5