# Plan — Phase A W1: 鲁棒性连续扫描（A1/A2）

日期：2026-09-20
对象：`results/20260926_robustness_sweep/`（规划产出）
关联：roadmap §2 Phase A W1；DATA_MANAGEMENT §10 评审流程（plan→review→execute）

## 1. 目标与假设

**目标**：把鲁棒性表述从"开/关"两点升级为**连续扰动幅度曲线**，替换论文 5.4 节，
并为 Phase D W13 的"真车退化 vs 仿真预测对照"提供数值表。

**假设（先写后验）**：
- H1：mean|e| 随扰动幅度**单调**增长（或出现明显拐点），非阶跃。
- H2：P+FF 基线的退化斜率 > RL，即 RL 的"域随机化训练"带来鲁棒性增益。
- H3：成功率直到较高扰动档位（≥40%）仍保持 10/10——因为先验（零动作=跟速）
  结构性保证了不追尾，退化主要表现为间距误差而非失稳。

## 2. 方法

**扰动定义**（对齐 env 的 DR 块，follow_env.py:123-124）：
- 名义值：τ=0.12 s，a_max=1.5 m/s²
- 扰动：**幅度 ε ∈ {0, 10, 20, 30, 40, 50}%**，方向取"退化"（执行器变差）：
  - τ 增大（电机更迟钝）：τ = 0.12·(1+ε)
  - a_max 减小（加速/减速更弱）：a_max = 1.5·(1−ε)
  - 这是"执行器整体退化"的最坏情形——比单向扰动信息量大。

> 语义说明：τ↑ 和 a_max↓ 都代表执行器变差（迟钝 + 无力），同向于"退化"。
> 若两参数都乘 (1+ε) 则混合了"迟钝+更猛"，不构成纯净退化，故不用。

**关键决策——不修改冻结环境**：`follow_env.py` 已 sha256 存证（铁律 2）。
按 roadmap A1 原话"或外层包装 env"，实现一个 **RobustnessWrapper**。

**已实测验证的注入点**（非推测）：
- `reset()` 内 tau/a_max 是**局部变量**直接传给 `Car`，外部无法传参注入；
- 但 `Car.tau` / `Car.a_max` 是**普通属性**，且 `Car.step()` **实时读取** `self.tau` /
  `self.a_max`（已读源码 + 实测 `step()` 正常）；
- 因此 wrapper 重写 `reset()`：先 `super().reset(seed)`，**随后立即**改写
  `self.follower.tau/a_max`（需要时也改 leader），再交还控制权。

```python
class RobustnessWrapper(gym.Wrapper):
    """Pin actuator params to the degraded regime for a robustness sweep.
    Environment file stays frozen; we only mutate Car attributes post-reset."""
    TAU0, AMAX0 = 0.12, 1.5
    def __init__(self, env, eps):          # eps: degradation fraction, 0..0.5
        super().__init__(env)
        self.eps = eps
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        e = self.eps
        self.env.follower.tau   = self.TAU0  * (1.0 + e)   # slower motor
        self.env.follower.a_max = self.AMAX0 * (1.0 - e)   # weaker accel
        return obs, info
```

注意：`evaluate()` 每 seed 都新建 env 并 `reset()`，故改写必须发生在 `reset()` 内
（而非构造时），否则被 `reset()` 覆盖。上面实现满足。

**与训练分布的对照**（重要）：env 的 DR 训练范围是
τ∈[0.08,0.18]、a_max∈[1.0,2.0]。本扫描的退化端点：
- ε=50% → τ=0.18（**恰为 DR 上界**）、a_max=0.75（**低于 DR 下界 1.0**）；
- 即 ε≤50% 档中，τ 始终在训练分布内，a_max 在 ε>33% 后超出训练分布。
- 因此本曲线同时包含**分布内泛化**（ε≤30%）与**分布外外推**（ε≥40%）两段——
  这正是论文要的"域随机化能否预测分布外退化"，须在图中分段标注。

**扰动方向（裁决 1）**：**首轮只跑"退化"单向**（τ↑、a_max↓），不做双向。
理由：RQ4 的物理含义是"真车执行器比模型**差**"；"偏快/偏灵"方向（τ↓、a_max↑）
大部分区域本就落在 DR 训练分布内，信息量低；而退化方向的 a_max 低端（<1.0）
恰是 DR 未覆盖处——曲线主要价值就在这段。时间预算 1 天，不翻倍。

**协议**：
- **R1（评审修订）——只退化 follower，leader 保持名义参数**：
  评估协议的公平性建立在"同一 seed = 同一 leader 行为"上。若 leader 也退化，
  其速度响应会变形，"退化来自谁"即混淆。RQ4 问的是**策略对自身执行器退化的
  鲁棒性**，leader 属环境，保持名义。→ wrapper **只改写 `env.follower`**，
  绝不触碰 `env.leader`。唯一变量是 policy 车的执行器。
- 评估对象：RL `ckpt/follow_stage2_final_v1.zip`（final 优于 best，报告 10.2 已证）
  与 `rule P+FF`（baseline_action use_ff=True）。
- 每档每策略：10 固定 seed（1000~1009，与论文一致）。
- **R2（评审修订）——主指标为无碰撞率，mean|e| 降级为副指标**：
  一旦某档出现撞车，回合提前终止，"稳态间距误差"在存活片段上计算**有偏**
  （越晚撞的回合均值越"好"）。故：
  - **主曲线**：collision-free rate（N/10）随 ε 变化；
  - **副指标**：mean|e| 仅在**无碰撞回合**上报告作参考，并注明样本数。
- **R4②（评审修订）——mean|e| 口径复用 `evaluate()` 的 settled 定义**
  （`train_ppo.py:88` 的末 80% 段），脚本直接调用该函数，避免两套口径。
- 全终止原因审计（铁律 5）：collision/lost/offtrack 逐档记录。
- **R4①（评审修订）——越界标注线用精确值 33.3%**（a_max = 1.5×(1−1/3) = 1.0，即 DR 下界处）。
- 输出：CSV 数值表 + 双曲线图（横轴 ε%，纵轴 collision-free rate；mean|e| 副图）。

## 3. 验收标准

1. `results/20260926_robustness_sweep/robustness_sweep.csv`：
   6 档 × 2 策略 × {collision-free rate, mean|e|(无碰撞回合), collision, lost, offtrack}；
2. `fig_robustness_curve.png`：主图 = collision-free rate vs ε%（RL 与 P+FF 两条）；
   副图 = mean|e|（仅无碰撞回合）vs ε%；
3. 图中在 ε=33.3% 处画**分布边界竖线**，标注"DR 内 / DR 外"；
4. **R3 试点门先于全档**（见下）；
5. RUNLOG 一条记录 + sha256 无环境改动（wrapper 独立文件，不进冻结集）。

## 3.5 R3 试点门（评审要求，全档前必过）

**只跑 ε=0% 一档**（wrapper 应完全透明），与已归档的 final 名义结果对照：

| 检查 | 期望 | 依据 |
|---|---|---|
| RL mean\|e\| | **0.93 cm**（两位小数一致） | `results/20260919_phase1_follow/eval_final_nominal.txt` |
| RL collision | 0 | 同上 |
| P+FF mean\|e\| | **1.93 cm** | 同上 |

- **一致** → wrapper 无侵入性，放行全档；
- **不一致** → 先修 wrapper，不烧全档机时。
- 该档同时验证 **R1**：若 leader 被误改，0.93 必然复现不出。

## 4. 时间预算

| 项 | 估计 |
|---|---|
| RobustnessWrapper + 评估脚本 | 0.5 天 |
| **R3 试点门（ε=0 单档）** | **~0.5 小时** |
| 6 档 × 2 策略 × 10 seed 评估 | ~1 小时（纯 CPU，follow 快） |
| 图表 + CSV + 归档 | 0.5 天 |
| **合计** | **~1 天 + 试点 0.5 小时** |

## 5. 风险与预案

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| wrapper 有侵入性（改变名义结果） | 低 | 全部数据无效 | **R3 试点门**先拦：ε=0 必须复现 0.93 cm |
| 某档 RL 碰撞率骤升（如 50% 档） | 中 | 曲线非单调，H3 被证伪 | 证伪也是结果；主指标为无碰撞率，恰能刻画失稳点 |
| P+FF 在低档位即撞车 | 中 | P+FF 曲线早断 | 撞车档标"crash"，保留原"开/关"两点作参照 |
| ε≥40% 档 a_max 越界导致行为异常 | 中 | 外推段不可解释 | 正是 RQ4 想要的 extrapolation 段；33.3% 处标边界线，分段解读 |

## 6. 关联

- 产出直接服务：论文 v2 §5.4（连续曲线）、Phase D W13 对照表（需要数值 CSV）。
- 数据归入 `results/20260926_robustness_sweep/`（DATA_MANAGEMENT §2 规范）。
- **评审记录**：`reviews/20260920_robustness_plan_review1.md`（§10 流程首次落地）。

## 7. 评审裁决与修订（已并入上文）

**三项裁决**：① 首轮只跑退化单向；② ε 封顶 50% 保持（33.3% 越界点是特性）；
③ 评估对象 follow_stage2_final 批准。

**四项修订**：R1 只退化 follower（leader 保持名义）；R2 主指标改无碰撞率、
mean|e| 降级为副（仅无碰撞回合）；R3 ε=0 试点门；R4① 越界线用精确 33.3%、
R4② mean|e| 复用 `evaluate()` 的 settled 口径。

**结论**：Approve with amendments，按 R1–R4 修改后执行。
