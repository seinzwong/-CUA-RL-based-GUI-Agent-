<h1 align="center">面向飞书桌面端差旅审批的 CUA 智能助理（RL-based GUI Agent）</h1>

本仓库聚焦飞书 CUA-Lark 赛题，采用 Step-RL 主干（Progress Estimator + Grounding Signal + PPO）构建办公自动化智能体。

## Quickstart (Feishu Default)

1. 一键跑通数据链路（探索 -> PRM -> RL 数据）：

```bash
bash eval/run_feishu_cua_pipeline.sh
```

2. 执行飞书评测：

```bash
bash eval/feishu_cua_eval.sh
```

3. 使用统一默认评测入口：

```bash
bash eval/run_eval.sh
```

更多快速说明见 [docs/quickstart_feishu.md](docs/quickstart_feishu.md)。

## 核心能力

1. 长链路办公任务建模：需求解析 -> 候选检索 -> 方案比较 -> 审批草稿 -> 提交。
2. 高可靠动作机制：提交前二次确认，支持回滚修订动作。
3. 使用 step-level 中间奖励进行稳定强化学习优化。

## 环境准备

```bash
conda create -n SPA python=3.9
conda activate SPA
pip install -r requirements.txt

conda create -n RL_train python=3.10
conda activate RL_train
pip install -r ppo/requirements.txt
```

## 使用流程

### 1) Base SFT

```bash
conda activate SPA
cd sft
python build_feishu_sft.py --input_file ../eval_agent/data/feishu_travel/train.json --output_file data/feishu_travel_sft.json --replicas 8
bash feishu_travel_qwen3_8b.sh
```

### 2) Exploration

```bash
cd ..
bash eval/run_feishu_cua_pipeline.sh
```

### 3) PRM 与 Stepwise Reward

```bash
python prm/data_org.py
python prm/inference_prm.py --input_data exploration/feishu_travel/exploration_outputs/exploration.json --output_file prm/exploration_inference_results_feishu_travel.json
python prm/rl_data_org.py
```

### 4) PPO

```bash
conda activate RL_train
bash ppo/train_ppo.sh
```

### 5) Evaluation

```bash
conda activate SPA
bash eval/run_eval.sh
```

## 文档

快速上手： [docs/quickstart_feishu.md](docs/quickstart_feishu.md)


## Citation

```bibtex
@article{wang2025spa,
  title={SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution},
  author={Wang, Hanlin and Leong, Chak Tou and Wang, Jiashuo and Wang, Jian and Li, Wenjie},
  journal={arXiv preprint arXiv:2505.20732},
  year={2025}
}
```
