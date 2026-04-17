# CUA-Lark 提交检查清单

## 1. 最小可演示资产
- [ ] 代码仓库可访问，默认入口为 Feishu CUA 路径
- [ ] 一键流程脚本可运行：eval/run_feishu_cua_pipeline.sh
- [ ] 评测脚本可运行：eval/feishu_cua_eval.sh
- [ ] 快速上手文档可读：docs/quickstart_feishu.md
- [ ] 结果模板可复用：docs/feishu_cua_result_template.md

## 2. 评审最关注的三句话
1. 业务问题：差旅审批是跨页面长链路任务，人工操作繁琐且有误提交风险。
2. 方法核心：保留 Step-RL + Progress Estimator + Grounding + PPO，迁移到飞书桌面 CUA。
3. 可靠性机制：提交前二次确认 + 回滚修订动作，降低高风险写操作误触发。

## 3. 演示前检查
- [ ] 模型服务可用（FastChat controller + worker）
- [ ] eval_agent/configs/task/feishu_travel.json 已生效
- [ ] eval_agent/envs/feishu_desktop_env.py 中安全约束逻辑正常
- [ ] 样例任务数据可读：eval_agent/data/feishu_travel/test.json
- [ ] 结果输出目录可写（默认 eval/feishu_cua_eval）

## 4. 演示流程话术（3分钟）
### 4.1 第 0-40 秒：问题定义
- 我们针对飞书桌面端差旅审批流程，解决长链路、跨页面、人工干预高的问题。

### 4.2 第 40-120 秒：方法与架构
- 架构保留 SPA-RL 主干：探索轨迹 -> PRM 进度估计 -> step-level reward -> PPO。
- 迁移重点在环境层与奖励语义：需求解析、候选检索、方案比较、审批草稿、提交确认。

### 4.3 第 120-180 秒：安全与可靠性
- 在提交动作前引入二次确认；若发现异常可触发 REVISE_FORM 或 CANCEL_SUBMIT 回滚。
- 这使得 Agent 从“会做”提升为“可控地做”。

## 5. 指标与证据准备
- [ ] Success Rate
- [ ] 平均步数 / 执行效率
- [ ] Safety Compliance（确认约束满足率）
- [ ] Rollback Recovery Rate（回滚后恢复成功率）
- [ ] Human Intervention Rate（人工介入比例）

建议统一填入 docs/feishu_cua_result_template.md 的表格中。

## 6. 常见问答（评审版）
### Q1: 你们是重新做了新算法吗？
A: 算法主干没有推倒重来。我们复用了 Step-RL/PRM/PPO，只替换环境交互与奖励语义，验证方法迁移能力。

### Q2: 为什么强调二次确认与回滚？
A: 办公场景包含高风险写操作，误提交成本高。二次确认和回滚机制是 CUA 可靠落地的必要条件。

### Q3: 当前是 Mock，如何落地真实飞书？
A: 已预留环境抽象接口，可后续接 PyAutoGUI 与飞书 OpenAPI，先用 PoC 验证策略闭环，再扩展生产集成。

### Q4: 与普通 RPA 的区别？
A: 我们不是固定脚本流程，而是用 RL + Grounding 学习长链路决策与中间奖励，具备更强泛化和异常恢复能力。

## 7. 交付前最后核对
- [ ] README 顶部出现 Feishu Quickstart
- [ ] run_eval.sh 默认指向 Feishu 评测
- [ ] 默认入口与脚本命名保持一致
- [ ] 关键脚本参数可配置，训练与评测路径保持一致
- [ ] 演示时使用统一口径：业务问题 -> 方法迁移 -> 安全可靠 -> 指标结果
