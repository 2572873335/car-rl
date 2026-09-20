# 「跟车」问题的 RL 解法：从零到 1.0 cm

> 用强化学习给仓储 AGV 车队做编队跟随，稳态间距误差 **1.00 cm**，
> 零碰撞；同观测、同接口、同随机种子下，最优手工规则是 **1.93 cm**。
> 独立复刻 11 项量化指标逐项一致。本文只讲一件事：**为什么安全约束应该
> 写进结构里，而不是靠奖励调参赌出来。**

---

## 一、先看结果

| 方案 | 稳态间距误差（名义参数） | 间距误差（执行器退化域随机化） | 碰撞 |
|---|---|---|---|
| 规则 P 控制 | 37.98 cm | 43.55 cm | **10/10** |
| 规则 P + 速度前馈 | 1.93 cm | 1.94 cm | 0 |
| **RL（PPO）** | **1.00 cm** | **1.06 cm** | **0** |

固定 10 个种子（1000–1009），同一套初始间距、同一条前车速度脚本、同一随机化序列。
RL 的逐种子误差落在 0.47–2.01 cm 之间，**逐种子全胜** P+FF——不是平均数效应。

任务本身很朴素：一台跟随车在双车道上跟着一台领头车，保持 20 cm 间距，
领头车按匀速 / 正弦 / 随机急刹三种模式变速。

## 二、手工规则的问题不是「调不出来」，而是「调对了也不解释」

上面表里最刺眼的一行是 P 控制的 **10/10 撞车**。它是个纯比例间距控制器，
在「追赶 + 前车变速」场景下 10 次试验全撞。而工程上加一行速度前馈就修好了：
P+FF 直接降到 1.93 cm、零碰撞。

**这恰恰暴露了规则系统的软肋**：性能完全取决于设计者有没有预见到这个工况。
未预见的工况就是软肋，而且软肋不会在测试里自己冒出来——你在验收时看到的是
1.93 cm 这个漂亮数字，看不到「如果前车刹车更急一点会怎样」。

所以本文的对照不是「RL 打赢了手调参数」，而是：**当工况分布可以被系统地度量
（域随机化）时，规则与学习策略的差异才能被量化，而不是靠设计者的完备性担保。**

## 三、核心方法：把安全写进结构里

这是我对 RQ1（如何把安全约束**结构性**嵌入而非依赖奖励调参）的答案。
三层，逐层收窄 RL 需要负责的范围：

### 1. 动作先验：零动作 = 与前车同速

不要让 RL 直接输出电机速度。**零动作意味着"保持当前速度"是既不安全、
也不含任何结构知识的默认值。** 改成在动作空间里嵌入先验：

```python
# follow_env.py
ACT_GAIN = 0.8   # v_cmd = v_f + ACT_GAIN * action
v_cmd = float(np.clip(v_l + ACT_GAIN * a, 0.0, V_MAX))
```

跟随车的速度指令是 `v_cmd = clip(v_leader + 0.8·a, 0, 1.3)`。领头车速度经车际
通信广播，于是 **`a = 0` 就等于与前车速度同步，几何上永不追尾**。
RL 只学一个残差修正量，而不是从零学"该开多快"。

这一步的副作用同样重要：P 控制、P+前馈这些规则方法可以写成
`a = f(s)` 的特例，**同一动作接口下做消融，天然公平**。

### 2. 奖励工程：只负责决策质量，不负责保命

安全已经由结构保证了，奖励就可以专心修信用分配。两条：

- **势能奖励整形（PBRS）**：`F = γΦ(s') − Φ(s)`，取 `Φ = −0.5·min(|e|, 1)`，
  `e` 为间距误差。理论保证不改变最优策略，效果是追赶阶段每缩短 1 cm
  立刻拿到奖励，不用等回合结束。
- **失败惩罚**：`−500·1_collision`，把撞车从"可以试试"变成确定性亏损。

跟车任务的逐步奖励是：

```
r = -(e/0.12)^2 - 0.3*(de/2.0)^2 - 0.5*(e_lat/0.15)^2 + F_t - 500*1_collision
```

**注意这三个分母（0.12 / 2.0 / 0.15）不是拍脑袋的**：它们分别对应
碰撞间距阈值、误差变化率的合理量级、横向偏差容差——量纲上是有物理含义的。

### 3. 动力学约束：物理已写明的事不让 RL 学

超车任务里，两条来自赛车物理的约束直接写进环境，对任何策略一视同仁：

```python
# overtake_env.py
R_min = self._min_turn_radius(lane_path, ..., window=0.8)
v_cmd = min(v_cmd, 0.8 * self.follower.w_max * R_min)   # 曲率感知限速
if self.switch_timer > 0:
    v_cmd = min(v_cmd, 0.6)                             # 换道后 0.8 s 瞬时限速
```

曲率前瞻 0.8 m 窗口内的最小转弯半径直接约束速度；换道执行后 0.8 s 内限速 0.6 m/s，
抑制横向机动期间的曲率失配超调。

**设计原则一句话：约束既缩小了搜索空间，又让安全由结构保证。**
RL 负责的是「什么时候换道、换道后推多快」这类决策，不是「物理上能不能做到」。

## 四、训练：先算更新次数，再看曲线

一个被反复忽视的量：

$$\text{梯度更新次数} = \frac{\text{总步数}}{n_{envs} \times n_{steps}}$$

我第一版跑了 50 万步，用的是 `32 环境 × 1024 步`——**只有 15 次更新**，
策略行为和随机无异。PPO 通常需要 300 次以上。改成 `n_steps=256`
（每轮 rollout 8192 步）、总步数提到 3–5 M 之后才正常。

本项目的三笔训练账：跟车 stage1 `2.5M/(32×256) = 305`；跟车 stage2 `610`；
超车微调 `610`。**全部 ≥ 300，这在本项目里是开训前的硬门槛。**

超参（全部实验一致）：PPO，MLP 128×128，lr 3e-4，n_steps 256 × 32 并行环境，
batch 512，γ=0.99，GAE λ=0.95，clip 0.2，熵系数 0.01。

## 五、超车：一个纯 RL 跨不过去的探索峡谷

超车任务的奖励地形是**双峰**的：跟随是安全的局部最优（回报 ≈ +65），
成功超车是全局最优（≈ +75 以上），两者之间隔着 −500 的撞车惩罚。

我两次从零开始训练，累计数千万步交互，**全部收敛到「永远跟随」**。
机制清楚：值函数学到了"靠近前车 = 危险"——探索噪声在间距 < 0.15 m 时导致撞车，
而它不知道「邻道并行在几何上不可能碰撞」（横向间距 0.15 m > 碰撞半径 0.12 m）。
安全超车需要先换道、再逼近、再反超，随机探索串不起这条完整动作链。
**回报梯度存在，但峡谷太深。**

解决方式不是继续加码奖励，而是**改变信息来源**：

| 步骤 | 结果 |
|---|---|
| 规则状态机采 300 局演示（41936 条转移） | — |
| 行为克隆热身（BC loss 2.0346 → 1.7721） | 2/10 成功、8 次碰撞 |
| PPO 微调 5 M 步 | **10/10 成功、零失误、1.4 s** |

BC 只有 2/10 成功，**这不重要**——它只要"会尝试"就够了，时机由 RL 接管后修复。
训练曲线上看得很清楚：回合长度先升后降（129 → 142 → 86），
上升段是 PPO 先学会"不撞车"（改为跟随，回合跑满），
下降段才学会超车（成功回合更短）。**先学安全、再学效率。**

对比手工规则状态机：RL 把平均超车耗时从 **2.0 s 压到 1.4 s（−30%）**，
平均车速 0.55 → 0.71 m/s。

还有一个意外发现：RL 学到的策略和两个基线都不同——**起步立即切入内圈并全程定居**。
规则状态机是"超完即回外圈"，这是人工设计的。RL 自己发现了内圈更短
（内圈 4.5424 m vs 外圈 5.4849 m，实测值）、且并行时横向间距 0.15 m > 碰撞半径 0.12 m，
几何上绝对安全——这个事实**没有任何一条规则显式编码过**。
（这两个长度我按 `overtake_env._paths()` 的实测值写；报告 §5.6 写的是
4.54 / 5.49 m，差异见溯源表「存疑 1」。）

## 六、诚实部分：RL 有失效边界，规则没有

把域随机化从「开/关」两点展开成连续曲线（执行器退化：τ 增大、a_max 减小），
对 RL 与规则 P+FF 各评估 **50 个固定种子**，主指标为无碰撞率：

| ε | τ (s) | a_max (m/s²) | RL 无碰撞 | RL mean\|e\| | P+FF 无碰撞 | P+FF mean\|e\| |
|---|---|---|---|---|---|---|
| 0% | 0.120 | 1.50 | 50/50 | 1.21 cm | 50/50 | 2.10 cm |
| 20% | 0.144 | 1.20 | 50/50 | 1.25 cm | 50/50 | 2.11 cm |
| 30% | 0.156 | 1.05 | **47/50** | 1.30 cm | 50/50 | 2.14 cm |
| 33% | 0.160 | 1.005 | **41/50** | 1.36 cm | 50/50 | 2.16 cm |
| 40% | 0.168 | 0.900 | **37/50** | 1.43 cm | 50/50 | 2.21 cm |
| 50% | 0.180 | 0.750 | **27/50** | 1.50 cm | 50/50 | 2.33 cm |

三点必须说清楚：

1. **RL 有失效边界，P+FF 没有。** RL 在 ε=30%（a_max=1.05，**仍在 DR 训练分布内**）
   就开始出现稀有失效（3/50），此后随 ε 增大单调加速恶化（3→9→13→23/50）。
2. **但 RL 的绝对精度全程更优**：即使 50% 外推档，RL 1.50 cm 仍优于 P+FF 的 2.33 cm
   （约 1.5 倍）。两种方法的鲁棒性画像互补——RL 更准但会失稳，P+FF 永不失效但不精确。
3. **分布内外分段**：ε≤20% 零失败；ε=33% 时 a_max=1.005，**仍 ≥ DR 下界 1.0，
   尚在分布内**；ε≥40% 才首次跌破 DR 下界（0.90 < 1.0）。

失效模式可解释：撞车全发生在大初始间距的追赶场景（gap0≈0.7–0.8 m），
此时 a_max 降到 0.75，**策略"指令所需的制动/加速物理上做不到"**。
这也直接回答了域随机化能预测什么：**它给的是「分布内」鲁棒性；分布外退化
不可由 DR 预测，且失效边界略早于分布外缘。**

## 七、复现性：11/11

这是我认为比 1.00 cm 更值钱的部分。

在**未修改环境与超参**的前提下，用统一脚本重跑了全部实验，逐项对照：

| 论文基准 | 复刻结果 | 差异 |
|---|---|---|
| 跟车 RL 1.00 cm（名义） | 1.00 cm | 0 |
| 跟车 RL 1.06 cm（DR） | 1.06 cm | 0 |
| 跟车 P+FF 1.93 cm | 1.93 cm | 0 |
| 跟车 P 撞车 10/10 | 10/10 | 0 |
| 超车 RL success 10/10 | 10/10 | 0 |
| 超车 RL 1.4 s | 1.4 s | 0 |
| 超车规则 2.0 s | 2.0 s | 0 |
| BC 2/10 成功 / 8 碰撞 | 2/10 / 8 | 0 |
| 超车 ep_len 129→140→86 | 129→142→86 | 峰值 +2 步 |
| 超车 ep_rew ≈ +72 | +72.5 | 0 |
| best 策略 0/10 跟随 | 0/10 跟随 | 0 |

**11 项逐项一致，唯一差异是训练曲线峰值差 2 步（1.4%）**，属随机波动。
能做到这一点的工程前提是**环境冻结**：`follow_env.py` / `overtake_env.py`
等 5 个核心文件以 sha256 存证，任何修改都会使哈希改变、必须从头重训。
零动作基线的表现也顺带验证了先验的价值：`a ≡ 0` 永不追尾（0 碰撞），
但永远跟着慢车（0/10 超车，平均车速 0.23 m/s 恰等于领头车速度）——
**先验负责生存，RL 负责决策**。

## 八、边界

- **仿真到现实的差距未实测。** 本文用二维运动学模型，迁移到实体 AGV 需要
  对电机时间常数、加速度上限做实测标定。域随机化接口已预留，但**没有实物闭环**。
- **感知层未涉及。** 观测是结构化状态，不含摄像头感知。
- **离线数据的边界。** 演示数据集（300 局 / 41936 条转移）由规则状态机采集，
  300/300 全 success——**是 positive-only 的，离线 RL 由此只能学到"好行为分布"，
  学不到失败规避。**

代码、模型与逐步复现指南全部开源。健康检查两条命令：

```bash
uv run python car_following_sim.py          # 跟车基线：mean|e| <= 1.2 cm
uv run python train_ot.py eval --rule-only  # 超车基线：10/10, 0 collision
```

---

## English Abstract

We reformulate fixed-gap car-following for low-speed AGV fleets as a reinforcement
learning problem in a 2D digital twin, and ask how safety constraints can be embedded
*structurally* rather than tuned through reward shaping. Our layered design — an action
prior where zero action already matches the leader's speed, residual learning on top of
it, and vehicle-dynamics constraints written into the environment — reduces steady-state
spacing error from 1.93 cm (best hand-tuned rule, P + speed feed-forward) to 1.00 cm
with zero collisions, using the identical observation/action interface and fixed seed sets.
A 50-seed continuous actuator-degradation sweep shows RL is more accurate at *every*
degradation level (1.50 cm vs 2.33 cm at ε=50%) but, unlike the rule baseline, develops a
failure boundary just *inside* the training distribution (47/50 collision-free at ε=30%),
so domain randomization buys in-distribution robustness only. An independent re-run of the
full pipeline matched 11 of 11 quantitative metrics, the sole deviation being a 2-step
difference in a training-curve peak. Code, models, and the 41,936-transition demonstration
dataset are released.

---

## 数字溯源表

> 规则：下表每个数字对应仓库中的真实文件与位置。命令类条目可直接复跑；
> 表末「存疑/需评议」列出我无法百分百对齐、建议重点核对的条目。

### 主结果

| 数字 | 出处文件 | 位置 |
|---|---|---|
| RL 1.00 cm / 1.06 cm，碰撞 0 | `results/20260919_phase1_follow/eval_best_nominal.txt`、`eval_best_dr.txt` | 各自汇总行 `RL(PPO) mean|e|= ...` |
| 规则 P 37.98 cm / 43.55 cm，碰撞 10/10 | 同上两个文件 | 各自汇总行 `rule P` |
| 规则 P+FF 1.93 cm / 1.94 cm | 同上两个文件 | 各自汇总行 `rule P+FF` |
| RL final 0.93 cm / 0.91 cm | `results/20260919_phase1_follow/eval_final_nominal.txt`、`eval_final_dr.txt` | 汇总行 |
| RL 逐种子 0.47–2.01 cm | `eval_best_nominal.txt` | 10 行 `settled|e|=` 明细 |
| 固定种子 1000–1009 | 上述 4 个文件 | 首行 `evaluating 10 fixed seeds` + 逐行 `seed=` |
| 表 1 三行口径 | `project_paper/project_report_full.md` | §10.2 表 1 复现对照 |
| 结论「提升约 2 倍」「逐 seed 全胜」 | `project_paper/project_report_full.md` | §5.2 三点解读 |

### 方法与代码

| 数字 | 出处文件 | 位置 |
|---|---|---|
| `ACT_GAIN = 0.8`，`v_cmd = clip(v_l + 0.8a, 0, 1.3)` | `follow_env.py` | 第 46 行常量 + 第 159 行 `step()` |
| 动作先验公式（式 11） | `project_paper/project_report_full.md` | §4.3 式 (11) |
| 跟车奖励式 | `follow_env.py` | 第 174–186 行；报告 §4.4 式 (13) |
| 奖励分母 0.12 / 2.0 / 0.15 | `follow_env.py` | 第 174 行附近 `reward = (-(e / 0.12) ** 2 ...)` |
| PBRS `Φ = −0.5·min(\|e\|,1)` | `follow_env.py` | 第 180–181 行 `phi = lambda x: -0.5 * min(abs(x), 1.0)` |
| 曲率限速 `v ≤ 0.8·w_max·R_min` | `overtake_env.py` | 第 205–207 行 |
| 换道瞬时限速 0.6 m/s / 0.8 s | `overtake_env.py` | 第 197 行 `switch_timer = 0.8`、第 210 行 `min(v_cmd, 0.6)` |
| 碰撞半径 0.12 m | `overtake_env.py` | 第 28 行 `COLLISION_2D = 0.12` |
| 车道间距 0.15 m | `overtake_env.py` | 第 44 行 `LoopPath(0.15, 1.50, 0.90, 0.15, y0=0.15)` |
| 域随机化 τ∈U(0.08,0.18)、a_max∈U(1.0,2.0) | `follow_env.py` | `reset()` 内 `tau = float(r.uniform(0.08, 0.18))` / `a_max = ...(1.0, 2.0)`；报告 §4.6 式 (16) |
| 控制步长 DT = 0.02 s | `car_following_sim.py` | 第 19 行 `DT = 0.02` |

### 训练与超参

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 跟车 stage1 305 次 / stage2 610 次 / 超车 610 次更新 | `results/20260919_phase1_follow/config.json`、`.../phase2_overtake/config.json` | `"updates_stage1"` / `"updates_stage2"` / `"updates"`；报告 §9.4 表 |
| 更新次数公式 | `project_paper/project_report_full.md` | §5.1「一个常被忽视的量」、§6.1 |
| 15 次更新（50 万步 / 32×1024） | `project_paper/project_report_full.md` | §6.1 案例一：`5×10^5/(32×1024) ≈ 15` |
| PPO 超参全表 | `results/20260919_phase1_follow/config.json` | `"hyperparams"` 对象 |
| 论文 stage1 2M 步 → 244 次更新（不足 300） | `DATA_MANAGEMENT.md` | §7 表下注 |
| BC loss 2.0346 → 1.7721 | `results/20260919_phase2_overtake/pretrain.log` | `BC epoch 1/10 loss=2.0346` … `BC epoch 10/10 loss=1.7721` |
| 演示集 300 局 / 41936 转移 | `results/20260919_phase2_overtake/pretrain.log` | `demo dataset: 41936 transitions` |
| BC 2/10、8 碰撞、2.2 s、0.50 m/s | `results/20260919_phase2_overtake/eval_bc.txt` | 汇总行 `RL(PPO) overtake= 2/10 ...` |
| 超车 RL 10/10、1.4 s、0.71 m/s | `results/20260919_phase2_overtake/eval_final_nominal.txt` | 汇总行 `RL(PPO) overtake=10/10 ...` |
| 规则状态机 10/10、2.0 s、0.55 m/s | 同上（及 `results/20260919_phase0_healthcheck/eval_overtake_ruleonly.txt`） | 汇总行 `rule-based` |
| 超车逐种子耗时 1.1–1.6 s | `eval_final_nominal.txt` | 10 行 `t_ot=` 明细 |
| ep_len 129 → 142(@0.39 M) → 86 | `results/20260919_phase2_overtake/train_finetune.log` | 首行 `ep_len_mean 129`、末行 `86.5`；峰值 142 见 `RUNLOG.md` Phase 2 / R2.3 |
| ep_rew −277 → +72.5 | 同上 | 首行 `ep_rew_mean -277`、末行 `72.5` |
| 跟随回报 ≈ +65 vs 成功 ≈ +75/+78 | `project_paper/project_report_full.md` | §4.4(c) 第 229–230 行「"永远跟随"的回报（≈+65）…劣于成功（≈+75）」、§4.5 第 242–243 行「（≈+75 以上）」、§6.2 第 619–620 行「低于成功（≈+78）」 |
| 数千万步从零探索全收敛到跟随 | `project_paper/project_report_full.md` | §4.5 第 244 行「累计数千万步交互均收敛到跟随策略」 |
| 零动作基线 0/10、0 碰撞、0.23 m/s | `results/20260919_phase2_overtake/eval_best_ot_verbose.txt` 汇总行；`project_paper/project_report_full.md` §5.3 表 2 | 见 §5.3 表 2「零动作（先验本身）」行 |

### 鲁棒性（50 seed）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 全部 14 行（ε / τ / a_max / 无碰撞 / mean\|e\|，RL 与 P+FF） | `results/20260926_robustness_sweep/robustness_sweep.csv` | CSV 全部数据行（policy, eps_pct, tau, a_max, n_seeds, collision_free, …, mean_abs_e_cm） |
| 表 5.5 呈现形式 | `project_paper/project_report_full.md` | §5.4.1 表 5.5 |
| 50 seed = 5 组 × 10 | `RUNLOG.md` | `[2026-09-20] Phase A W1 补充 — 50-seed 复跑` 命令 `--seeds 1000..1009 2000..2009 3000..3009 4000..4009 5000..5009` |
| 失效全在大初始间距追赶场景、gap0≈0.7–0.8 m、a_max=0.75 | `project_paper/project_report_full.md` | §5.4.1「失效模式可解释」段 |
| ε=33% 时 a_max=1.005 仍在分布内 | `results/20260926_robustness_sweep/robustness_sweep.csv` | ε=33 行的 `a_max` = 1.0050 |
| 30/30 的 Wilson 95% 下界≈89% | `project_paper/project_report_full.md` | §4.8.4 第 1 点 |

### 复现性与工程纪律

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 11/11 逐项对照表 | `project_paper/project_report_full.md` | §12.1 表 |
| 唯一差异：峰值 142 vs 140（+2 步，1.4%） | `project_paper/project_report_full.md` | §12.1 末 + §12.3「1 项曲线峰值在随机波动内」 |
| 5 个冻结节文件 sha256 | `DATA_MANAGEMENT.md` | §8 哈希表 |
| 健康检查 1.10 cm / 规则 10/10 | `results/20260919_phase0_healthcheck/sim_follow_baseline.txt`、`eval_overtake_ruleonly.txt` | 各文件正文 |
| 演示集 300/300 success、positive-only | `demos_v1_README.md` | 「用途与边界」 |
| `demos_v1.npz` sha256 | `DATA_MANAGEMENT.md` | §9 数据资产哈希表 |

### 内圈/外圈长度（自行复算，见存疑项）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 内圈 4.5424 m、外圈 5.4849 m、比值 0.8282 | 由 `overtake_env._paths()` 实测 | 复算命令：`.venv/bin/python -c "from overtake_env import _paths; o,i=_paths(); print(o.length, i.length)"` |
| 报告版「4.54 m vs 5.49 m」 | `project_paper/project_report_full.md` | §5.6 第 571 行、§11.5 第 1018 行 |

### 存疑 / 需评议（重点核对这三条）

1. **内圈长度 4.5424 m vs 报告 §5.6 的 4.54 m**：4.5424 四舍五入为 4.54，
   一致。但**报告 §5.6 同时写「外圈 5.49 m」，而我用同一对象实测得的
   `_paths()[0].length` = **5.4849 m**，四舍五入应为 5.48。解析几何手算
   （4 段直线 3.6 m + 4 段 1/4 圆弧 2π·0.3 = 1.88496 m）= 5.48496 m，
   **与 5.48 一致、与报告的 5.49 差 1 cm**。本文正文因此采用实测值 5.4849，
   **未采用报告的 5.49**。若报告要以 5.49 为准，需说明舍入口径。
2. **「14 轮」的出处**：唯一出处在 `project_paper/project_report_full.md`
   （§6 开头「表 3 汇总本文 14 轮迭代」+ 表 3 恰有 14 行，行号 588–611）。
   **`RUNLOG.md` 的条目数是 22**（16 条 `### [2026-09-19] R*` + 6 条
   `## [2026-09-20]` 日期条目），与 14 不是同一个计数对象——RUNLOG 记的是
   「运行」，表 3 记的是「失败—诊断—修复」事件。更关键的是：**表 3 的 14 行
   中，多数事件发生在本仓库 RUNLOG 开始记录之前**（属论文 v1 开发期的
   迭代），因此这 14 行**在仓库内没有逐条对应的原始日志**。本文保留「14」
   （因为它是表 3 的真实行数），并在正文/溯源中如实说明其性质。
3. **「评审流程已拦截 4 次真问题」**：出处为 `AGENT_HANDOFF.md` §5 末。
   该句的括号内只列了 3 项（论文三轮 19 项 + sb3-contrib 库选型错误 +
   报告 §4.8 的 30-seed 溯源缺失），**与其自身声称的「4 次」数目不符**。
   我可以逐条具名引证的拦截是：(a) sb3-contrib 不含离线 RL 算法 →
   `research/roadmap.md` §A4 行 + commit `1e6b786`；(b) W2 plan §3 数据映射
   错误 → `reviews/20260920_w2_plan_review1.md` R1；(c) d3rlpy pin
   `gymnasium==1.0.0` 会降级主环境、必须独立 venv → 同文件「评审评价」段 +
   `requirements-offline.txt` 头注释；(d) 报告 §4.8 的 30-seed 数字无命令溯源
   → `reviews/20260920_report48_review1.md` 发现 #1。评审记录内部的序号
   （W2 review 自称「第二次」、report48 review 自称「第四次」）与
   AGENT_HANDOFF 的清单同样对不齐。**建议以具名四条的版本对外表述。**

4. **ε=33% 档 a_max 的显示值不一致（小）**：本文表格采用
   `robustness_sweep.csv` 的原始值 `1.0050`（正文写 1.005）；
   报告 §5.4.1 表 5.5 该格显示为「**1.00（DR 下界）**」。
   两者是同一数据的舍入/显示差异，不影响「仍在 DR 分布内（≥1.0）」的判定
   （DR 下界为 1.0，见 `follow_env.py` `a_max = float(r.uniform(1.0, 2.0))`）。
   建议统一为 `1.005`。

---

> 草稿版本：v1（未发布）。项目仓库：`/home/zy/car_rl/code0919`，
> 全部数字来自 `results/` 原始输出，可 `make check` / `make reproduce` 复跑。
