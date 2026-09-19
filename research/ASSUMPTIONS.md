# ASSUMPTIONS.md — 假设登记表

> 规范：**任何训练/实验开跑前**在此登记假设与判定阈值；结果出来后对照检验。
> 阈值训练前定死；结果落在边界则补 seed 确认，**不放宽阈值**。
> 依据：`DATA_MANAGEMENT.md` §10（plan → review → execute）。

---

## [2026-09-20] RQ2 离线 RL 对比（Phase A W2）

**登记时间**：训练启动前（`rq2_offline.py train` 已派出）
**Plan 依据**：`plan_W2_offline_rl.md`（review1，Approve with amendments）
**数据**：`demos_v1.npz`（300 集 / 41936 转移，全 success）

| # | 假设 | 判定阈值（定死） | 登记时状态 |
|---|---|---|---|
| H1 | IQL 成功率**介于** BC(2/10) 与 PPO(10/10) 之间 | IQL ∈ **[3/10, 9/10]** | 未测 |
| H2 | IQL 碰撞数 **< BC 的 8 次** | collision **≤ 6/10** | 未测 |
| H3 | TD3+BC 与 IQL **同档** | 成功率差 **≤ 2/10** 且碰撞差 ≤ 2 | 未测 |

**验证协议**：10 固定 seed（2000–2009）、deterministic、名义 + DR 两组、
五类终止原因全审计（铁律 5）。

---
