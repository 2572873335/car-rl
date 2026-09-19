# Plan — Phase A W2: RQ2 离线 RL 对比（A4/A5/A6）

日期：2026-09-20
对象：`results/2026092x_rq2_offline/`（规划产出）
关联：roadmap §2 Phase A W2（RQ2）；`DATA_MANAGEMENT.md` §10 评审流程；
     W1 review（`reviews/20260920_robustness_plan_review1.md`）及其三条收尾意见

## 0. 本轮要回答什么

**RQ2**：同一批演示数据下，模仿（BC）、离线 RL、在线微调三者的
**数据效率与安全边界**如何排序？

**主结论表（四方对照）**：BC（已有 2/10）/ IQL / TD3+BC / PPO 微调（已有 10/10），
同种子、同指标。

## 1. 假设登记（先写后验，进 research/ASSUMPTIONS.md）

| # | 假设 | 若成立的含义 | 若被证伪的含义 |
|---|---|---|---|
| H1 | IQL 成功率落在 BC(2/10) 与 PPO(10/10) **之间** | 离线 RL 能从纯演示中提取超出模仿的策略 | 离线 RL 未超越模仿 → 数据覆盖不足 |
| H2 | IQL 碰撞数 **< BC 的 8 次** | 离线 RL 的保守性（Q 下界）抑制撞车 | 保守性未体现，需查实现/超参 |
| H3 | TD3+BC 与 IQL **同档**（成功率相差 ≤2/10） | 换算法不影响结论，结论稳健 | 结论算法依赖，需报告两者差异并解释 |

> 登记表同时在 `research/ASSUMPTIONS.md` 留档，**训练前**写入。

## 2. 算法与库选型（**已实测确认，非推测**）

- **d3rlpy 2.8.1**（独立 venv `/tmp/d3check` 已装并验证；版本锁 2.8.1）
- **主**：`IQLConfig`、`TD3PlusBCConfig`；**可选第三**：`CQLConfig`（均已确认存在）
- API 形态（实测）：`IQLConfig(...).create(device=...).fit(dataset, n_steps=...)`
- **R3 式试点门（W2 版）**：先跑 **100 transition 微型拟合**验证
  `MDPDataset` 构造 + `fit` 跑通、再开全量。避免 API 细节（2.x breaking changes）
  浪费全量机时。

**环境隔离（关键）**：d3rlpy pin `gymnasium==1.0.0`，会降级本项目冻结的 1.3.0。
→ **绝不在主 `.venv` 装 d3rlpy**。全部离线 RL 训练在独立 venv（如 `.venv-d3rlpy`）
执行，**只读** `demos_v1.npz`，产出模型存 `ckpt_offline/`。评估回到主 venv
（用 SB3 的 eval harness，或加载 d3rlpy 模型单独评估）。

## 3. 数据映射（映射 A：goal-as-terminal）

按 `export_demos.py` docstring 的**映射 A** 构造 `MDPDataset`：

```
terminals  = terminated | success      # 真 MDP 终态（含成功）
timeouts   = truncated | timeout | failed  # 时间限制截断（非真终态）
```

**理由**：成功是真·终态（任务完成），超时/失败是时间限制截断，二者必须区分——
否则 Q 值会在"超时"处错误地认为后续无价值，污染回报估计。
`export_demos.py` 已按此语义输出 `terminals`（terminated|truncated）与
`timeouts`（truncated）；**构造时需重算**：`terminals = (reason != timeout/failed)`。

字段对应（实测 MDPDataset 签名）：
`MDPDataset(observations, actions, rewards, terminals, timeouts=...)`。

## 4. 评估协议（**与论文完全对齐——评审重点盯此条**）

- **固定 10 seed（2000–2009）**，与论文表 2 / W1 同种子；
- **deterministic** 策略评估；
- **名义 + 域随机化两组**；
- **五类终止原因全审计**（collision/lost/offtrack/success/failed，铁律 5）；
- 指标同表 2：成功率 / 碰撞 / 冲出 / t_overtake / mean_v；
- **离线 RL 不得在任何训练环节接触环境**——只在评估时接触。
  这是"离线"叙事成立的底线；plan 中声明，实现中以代码结构强制
  （训练脚本不 import env，评估脚本单独）。

## 5. 对照表设计（论文 4.8 节全部素材）

| 方法 | 训练方式 | 数据 | 成功率 | 碰撞 | 冲出 | t_ot | mean_v |
|---|---|---|---|---|---|---|---|
| BC（已有） | 监督 | 41936 演示 | 2/10 | 8 | — | — | — |
| IQL | 离线 | 同上 | ? | ? | ? | ? | ? |
| TD3+BC | 离线 | 同上 | ? | ? | ? | ? | ? |
| PPO 微调（已有） | 在线 | BC 起点 +5M | 10/10 | 0 | 0 | 1.4 s | 0.71 |

**逐 seed 列 t_ot**（评审要求：这张表就是 4.8 节全部素材）。

## 6. 风险与预案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| d3rlpy 连续动作稳定性差 | 中 | 训练不收敛 | 降 n_steps + 观察评估曲线早停 |
| IQL 全部撞车（极端） | 低 | 结论负面 | 如实记录，这是"离线数据覆盖不足"的诊断；换 BEAR/BCQ 复测 |
| 环境隔离失败（误装主 venv） | 低 | 冻结环境被破坏 | 训练脚本开头 assert `d3rlpy` 不在主 venv；单独 venv 运行 |
| API 细节错（2.x breaking） | 中 | 机时浪费 | **100-transition 试点门**先验证 |
| 单点比较答不了"数据效率" | 中 | RQ2 未答 | **见下：数据量扫描** |

## 7. 数据效率扫描（对 RQ2 的强化，评审 W1 已提）

RQ2 问的是"**数据效率**"，单点(300 demos)只能答"谁最好"。增加 demo 数量扫描：
**50 / 100 / 200 / 300 demos** × {BC, IQL, TD3+BC} → 成功率曲线。
成本低（BC 秒级、IQL 分钟级），把四方表升级为"数据量 × 方法"矩阵，
才真正回答 RQ2。若时间紧，此项降级为"仅 300 档 + 100 档"两点。

## 8. W1 收尾三条修订（评审要求，并入本轮）

1. **A6 论文措辞同步**：现有"域随机化下性能完全不退化"必须加限定
   "**在训练分布范围内**"，并在 5.4 新增分布外失效边界发现（W1 数据）；
2. **50-seed 置信**：W1 已用 50 seed 重跑（RL: 0/10/20%→50/50，30%→47/50，
   33%→41/50，40%→37/50，50%→27/50；P+FF 全程 50/50）——已落盘，用于论文；
3. **互补鲁棒性论点**：写进 5.4 讨论段与展望——RL 精度优（1.50 vs 2.33 cm）、
   P+FF 零碰撞，二者画像互补 → 未来 work：给 RL 输出加"物理可行性护栏"
   （a_max 不足时禁止超可行制动/加速指令），理论上兼得精度与安全。

## 9. 时间预算

| 项 | 估计 |
|---|---|
| d3rlpy 独立 venv + 100-transition 试点门 | 0.5 天 |
| 数据映射 + 训练脚本（IQL/TD3+BC） | 0.5 天 |
| 四方评估（10 seed × 2 组 × 3 方法）+ 归档 | 1 天 |
| 数据量扫描（50/100/200/300） | 0.5 天 |
| 论文 v2 §4.8 草拟 + 5.4 修订 | 1 天 |
| **合计** | **~3.5 天（W2 一周内余量充足）** |

## 10. 关联

- 输入：`demos_v1.npz`（41936 转移，sha256 已入 DATA_MANAGEMENT §9）
- 输出：`results/2026092x_rq2_offline/`（CSV + 表 + 图）
- 假设登记：`research/ASSUMPTIONS.md`
- 论文素材：§4.8 新增 + §5.4 修订（A6）

## 待评审确认点

1. 算法：IQL + TD3+BC 为主，CQL 可选第三 —— 是否够？
2. 数据量扫描（§7）是否本轮做，还是降级为两点？
3. 评估 harness：复用主 venv 的 SF3 脚本加载 d3rlpy 模型，还是 d3rlpy 侧单独写
   评估（需要跨 venv 的模型序列化考虑）？
