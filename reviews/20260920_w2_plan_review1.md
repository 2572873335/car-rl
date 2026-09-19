# 评审记录 — Phase A W2 离线 RL Plan

- **日期**：2026-09-20
- **对象**：`plan_W2_offline_rl.md`（Phase A W2：RQ2 离线 RL 对比）
- **评审轮次**：review1
- **流程依据**：`DATA_MANAGEMENT.md` §10（plan → review → execute）

---

## 评审评价

> W2 plan 整体高质量。**环境隔离那条（d3rlpy 会 pin 降级 gymnasium 1.0.0）是
> 又一个"动手前拦截"级的发现**——若直接装进主 venv，冻结环境的 1.3.0 被悄悄
> 降级，前面所有哈希存证就形同虚设。

## 裁决（Approve with amendments）

### R1（必修，**拦截级**）——§3 数据映射描述错误

Plan 原文称 `export_demos.py` "已输出 `terminals`（terminated|truncated）
与 `timeouts`（truncated）"，不能直传 npz 字段给 `MDPDataset`。

**事实澄清（复刻者实测核实）**：评审结论**正确**，但对 npz 的描述需精确化——
- npz **确实有** `terminals` / `timeouts` 字段（不是"不存在"）；问题是**语义不对**：
  `timeouts` 字段 = 300（因 env 把 `success` 报成 `truncated=True`），
  而映射 A 要求成功集 `timeouts=0`；
- npz **没有** 逐步 `terminated` / `truncated` / `success` 原始数组（评审称"8 个
  原始数组、三者分开"亦不准确）；
- 正解：从 `episode_reasons` 重构（实测 → terminals=300, timeouts=0）。

→ **直接照 plan 原稿写代码必炸**（KeyError 或语义错误），修订 §3 后再执行。

### R2（Q3 裁决）——评估放 d3rlpy venv 内，加规则基线锚点门

不跨 venv 序列化（工程量不值）。评估脚本在 d3rlpy venv 内直接
`import overtake_env`（源码共享、解释器不同）。
**锚点门**：先在 d3rlpy venv（gymnasium 1.0.0）跑规则基线评估，
须复现 **10/10、2.0s**。复现 → 两版本对本 env 行为等价，继续；
否则回退 state_dict 导出方案。与 ε=0 门同模式。

### R3（Q1 裁决）——IQL + TD3+BC 足够

CQL 仅在"确认支持连续动作"前提下做第三算法，**限时 30 分钟**，查不到即丢。

### R4（Q2 裁决）——数据量扫描做全 4 点

50/100/200/300 × {BC, IQL, TD3+BC}。**每点 BC 必须重新训练**，不得复用
现有 300-demo 的 BC 结果。

### R5（小项）——假设登记表加"判定阈值"

H1：IQL ∈ [3/10, 9/10] 算成立（留随机波动带）；
H2：collision ≤ 6/10；
H3：成功率差 ≤ 2/10 且碰撞差 ≤ 2。
避免事后"差不多就算"。

## 结论

**Approve with amendments** —— 按 R1–R5 修订后执行。
顺序：改 plan §3/R5 → **两个试点门并行**（100-transition API 门 + d3rlpy venv
规则基线锚点门）→ 全量训练 → 数据量扫描 → 评估归档。

## 流程价值

这是评审流程**第二次在动手前拦截真问题**（首次 = sb3-contrib 无离线 RL）。
R1 若带入执行，会在 `MDPDataset` 构造处炸出，且排查方向大概率被误导到库版本上。

---

## 响应

R1–R5 全部接受。§3 已按**实测事实**重写（并附事实澄清：评审结论对、
描述需精确化）；§1 加判定阈值列与 R2/R3/R4 裁决；§11 裁决汇总；§12 更新时间预算。

**注**：R1 的实测澄清本身即是"plan 描述须经事实验证"这一纪律的又一例证。
