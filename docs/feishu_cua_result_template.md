# 飞书 CUA-Lark 实验结果模板（路演/简历复用）

## 1. 业务问题
- 场景: 飞书桌面端差旅审批长链路
- 痛点: 跨页面操作繁琐、人工确认成本高、误提交风险高
- 目标: 低干预、高可靠地完成需求解析 -> 候选检索 -> 方案比较 -> 审批草稿 -> 提交

## 2. 方法概览
- Agent范式: CUA / GUI Agent
- 训练框架: Step-RL + Progress Estimator + Grounding Signal + PPO
- 模型底座: Qwen3-8B（可替换）
- 安全策略: 提交前二次确认、支持回滚修订动作

## 3. 实验设置
- 环境: FeishuDesktopEnvMock
- 任务集: eval_agent/data/feishu_travel/test.json
- 关键动作: OPEN_TRAVEL_CENTER, PARSE_TRAVEL_REQUEST, SEARCH_CANDIDATE_OPTIONS, COMPARE_OPTIONS, FILL_APPROVAL_FORM, GENERATE_APPROVAL_DRAFT, CONFIRM_SUBMIT, SUBMIT_APPROVAL
- 评测命令: bash eval/feishu_cua_eval.sh

## 4. 指标定义
- Task Success Rate: 任务成功率
- Step Efficiency: 完成任务平均步数
- Safety Compliance: 提交前确认约束满足率
- Rollback Recovery Rate: 回滚后恢复成功率
- Human Intervention Rate: 人工介入比例

## 5. 结果填写（示例表）
| 指标 | 基线Agent | CUA-RL Agent | 提升 |
|---|---:|---:|---:|
| Task Success Rate |  |  |  |
| Step Efficiency (lower better) |  |  |  |
| Safety Compliance |  |  |  |
| Rollback Recovery Rate |  |  |  |
| Human Intervention Rate (lower better) |  |  |  |

## 6. 关键案例
- 案例A: 需求解析 + 方案比较 + 草稿生成一次通过
- 案例B: 检测风险后触发 REVISE_FORM, 最终成功提交
- 案例C: 缺少二次确认时阻止 SUBMIT_APPROVAL

## 7. 对飞书场景的价值
- 审批协同: 复杂审批链路自动化执行
- 办公自动化: 减少重复输入与页面跳转
- 高可靠执行: 动作校验 + 约束执行降低误操作
- 可扩展性: 可接入 PyAutoGUI / 飞书 OpenAPI 形成生产闭环

## 8. 简历表述建议（可直接复用）
- 基于 Step-RL 与 Grounding Signal，将移动端长链路Agent迁移至飞书桌面端 CUA 场景，完成差旅审批 PoC。
- 构建需求解析-候选检索-方案比较-审批草稿生成任务链，并引入提交前二次确认与回滚机制，实现低干预高可靠执行。
- 在统一训练范式下完成 PRM 奖励塑形与 PPO 强化优化，验证 RL 方法在办公 GUI 任务中的迁移能力。
