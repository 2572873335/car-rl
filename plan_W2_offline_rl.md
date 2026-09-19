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

**R5 修订——判定阈值前置**（避免事后"差不多就算"的弹性解释）：

| # | 假设 | 判定阈值（训练前定死） | 若成立 | 若被证伪 |
|---|---|---|---|---|
| H1 | IQL 成功率**介于** BC(2/10) 与 PPO(10/10) 之间 | IQL ∈ **[3/10, 9/10]** | 离线 RL 从纯演示提取超出模仿的策略 | 未超越模仿 → 数据覆盖不足 |
| H2 | IQL 碰撞数 **< BC 的 8 次** | collision **≤ 6/10** | 离线 RL 保守性抑制撞车 | 保守性未体现，查实现/超参 |
| H3 | TD3+BC 与 IQL **同档** | 成功率差 **≤ 2/10** 且碰撞差 ≤ 2 | 换算法不影响结论，结论稳健 | 结论算法依赖，报告差异并解释 |

> 登记表同时在 `research/ASSUMPTIONS.md` 留档，**训练前**写入。
> 阈值训练前定死；结果落在边界则**补 seed 确认**，不放宽阈值。

**R3 裁决——算法范围**：IQL + TD3+BC **足够**。CQL 仅在"确认支持连续动作"
前提下做第三算法，**限时 30 分钟**，查不到即丢。

**R4 裁决——数据量扫描做全 4 点**（50/100/200/300 × {BC, IQL, TD3+BC}）。
**约束：每个数据点的 BC 必须重新训练**（不得复用 300-demo 的 BC 结果）。

**R2 裁决——评估在 d3rlpy venv 内做，加"规则基线锚点门"**：
不跨 venv 序列化。评估脚本在 d3rlpy venv 内直接 `import overtake_env`
（源码共享、解释器不同）。**锚点门**：先在 d3rlpy venv（gymnasium 1.0.0）里
跑**规则基线**评估，须复现 **10/10、2.0s**——复现则两版本对本 env 行为等价；
否则回退 state_dict 导出方案。

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

**R1 修订（评审拦截级）——npz 实际字段（已实测核实，非推测）**：

```
observations(41936,6) actions(41936,2) rewards(41936,)
next_observations(41936,6)
terminals(41936,) bool   = 逐步 (terminated | truncated)
timeouts(41936,)  bool   = 逐步 truncated
episode_ends(300,) episode_reasons(300,) episode_seeds(300,)
episode_terminals(41936,) bool
```

**关键陷阱（必须重算，不能直传）**：本 env 把 `success` 报成
`truncated=True`（`overtake_env.py:265`），因此
- npz 的 `timeouts` 字段 = **300**（每集末尾一个 True），
- 而映射 A 要求成功集 `timeouts=0`。

→ **直传 npz 的 `terminals`/`timeouts` 给 `MDPDataset` 是错的**。
（注：npz **未**单独存逐步 `terminated`/`truncated`；原始信息只在
`episode_reasons` 里，故必须由此重构。）

**正确构造（已实测，产出 terminals=300 / timeouts=0）**：

```python
TRUE_TERM = {"collision", "offtrack", "lost", "success"}   # 映射 A：成功=真终态
terminals = np.zeros(n, bool)
timeouts  = np.zeros(n, bool)
for end, reason in zip(d["episode_ends"], d["episode_reasons"]):
    if str(reason) in TRUE_TERM:
        terminals[end] = True
    else:                      # timeout / failed = 时间限制截断
        timeouts[end] = True
dataset = MDPDataset(observations, actions, rewards, terminals, timeouts=timeouts)
```

**理由**：`collision/offtrack/lost` 是坏的终止，`success` 是目标达成的真终态；
`timeout/failed` 才是时间限制截断。二者在 Q 值 bootstrap 上处理不同，
必须区分，否则回报估计被污染。
字段顺序与实测 `MDPDataset` 签名一致：
`MDPDataset(observations, actions, rewards, terminals, timeouts=...)`。

> **附带改进项（可选）**：`export_demos.py` 采集时算了 `term_buf`（逐步
> terminated）但未存入 npz。若将来采集含失败轨迹的数据集，应补存原始
> `terminated`/`truncated` 以免依赖 `episode_reasons` 重构。当前数据集
> （全 success）不受影响。**本轮不改**（改则须重跑 + 重算 sha256）。

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

## 待评审确认点（**已裁决，见 §1 / §11**）

1. ~~算法~~ → **R3：IQL + TD3+BC 足够**（CQL 限时 30 分钟可选）
2. ~~数据量扫描~~ → **R4：做全 4 点**，每点 BC 重训
3. ~~评估 harness~~ → **R2：d3rlpy venv 内评估 + 规则基线锚点门**

## 11. 评审裁决汇总（review1，2026-09-20）

| 项 | 裁决 | 处理 |
|---|---|---|
| R1 | §3 数据映射描述错误（**拦截级**） | 已实测核实并重写 §3（见下"R1 事实澄清"） |
| R2 | 评估放 d3rlpy venv + 锚点门 | 已写入 §1 / §3 |
| R3 | IQL + TD3+BC 足够 | 已写入 §1 |
| R4 | 数据量扫描全 4 点 | 已写入 §1 / §7 |
| R5 | 假设加判定阈值 | 已写入 §1 |

**R1 事实澄清（实测）**：评审 R1 的**结论正确**（不能直传 npz 字段），
但其对 npz 的描述需精确化——npz **确实有** `terminals`/`timeouts` 字段，
只是**语义不对**（`timeouts`=300，因 env 把 success 报成 truncated）；
npz **没有** 逐步 `terminated`/`truncated`/`success` 原始数组。
正解：从 `episode_reasons` 重构（已实测 → terminals=300, timeouts=0）。
详见 §3。

**结论**：Approve with amendments，按 R1–R5 修订后执行。
顺序：改 plan §3/R5 ✅ → **两个试点门并行**（100-transition API 门 +
d3rlpy venv 规则基线锚点门）→ 全量训练 → 数据量扫描 → 评估归档。

**流程价值**：这是评审流程第二次在动手前拦截真问题（首次= sb3-contrib 无离线 RL）。
R1 若带入执行，会在 `MDPDataset` 构造处炸出，且排查方向大概率被误导到库版本上。

## 12. 时间预算（更新）

| 项 | 估计 |
|---|---|
| d3rlpy 独立 venv + **两个试点门** | 0.5 天 |
| 数据映射 + 训练脚本（IQL/TD3+BC） | 0.5 天 |
| 四方评估（10 seed × 2 组 × 3 方法）+ 归档 | 1 天 |
| 数据量扫描（50/100/200/300 × 3 方法，BC 重训） | 1 天 |
| 论文 v2 §4.8 草拟 + 5.4 修订 | 1 天 |
| **合计** | **~4 天（W2 一周内）** |
