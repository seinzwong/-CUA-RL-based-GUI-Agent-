# Feishu CUA-Lark Quickstart

## 30秒读懂
这是一个面向飞书桌面端差旅审批流程的 CUA (Computer-Use Agent) PoC。

它保留了原项目的 Step-RL 技术主干（Progress Estimator + Grounding Signal + PPO），
但将业务场景从电商迁移到办公自动化，并加入高风险写操作安全约束：
- 提交前二次确认
- 支持回滚修订（REVISE_FORM / CANCEL_SUBMIT）

默认入口已统一到 Feishu 方向。

## 3条命令复现主流程

1. 一键生成 Feishu 流程数据并组织 RL 训练数据：

```bash
bash eval/run_feishu_cua_pipeline.sh
```

2. 启动 Feishu CUA 评测：

```bash
bash eval/feishu_cua_eval.sh
```

3. 使用统一默认入口评测（已映射到 Feishu）：

```bash
bash eval/run_eval.sh
```

## 关键目录
- eval_agent/envs/feishu_desktop_env.py: 飞书桌面差旅审批 Mock 环境（含二次确认/回滚）
- eval_agent/tasks/feishu_travel.py: 飞书任务定义
- eval_agent/configs/task/feishu_travel.json: 飞书评测配置
- exploration/feishu_travel/: 飞书探索轨迹采集
- prm/: 进度估计、推理、RL 数据组织（默认飞书路径）
- ppo/: PPO 训练入口（默认飞书 RL 数据）

## 简历一句话版
基于 Step-RL 与 Grounding Signal，将长链路 Agent 从电商迁移到飞书桌面端 CUA 场景，构建差旅审批 PoC，并通过提交前二次确认与回滚机制实现低干预高可靠执行。

## 路演汇报模板
详见 docs/feishu_cua_result_template.md
