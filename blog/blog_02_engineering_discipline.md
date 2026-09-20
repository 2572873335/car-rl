# 14 轮 debug 教会我的 RL 工程纪律

> 一个把「仓储 AGV 编队跟随」做成强化学习项目的复盘。不聊算法，
> 只聊 14 轮「失败 → 诊断 → 修复」里真正咬人的坑，以及一条
> **已拦截 4 次真问题的流程**——包括拦下我自己的一次数据诚信瑕疵。

## 先说「14 轮」的出处

标题里的 14 不是修辞。项目报告里有张表叫「表 3　十四轮『失败—诊断—修复』迭代」，
**那张表恰好 14 行**，逐行是「症状 / 根因 / 修复」。

但口径要说清：另一个计数对象 `RUNLOG.md` **有 22 条记录**（16 条 `R*` 运行条目
+ 6 条阶段条目）。**RUNLOG 记的是「运行」，表 3 记的是「事件」**；且表 3
多数事件发生在 RUNLOG 开始逐条记录之前，**没有逐条对应的原始日志**。

---

## 坑 1：把「手写汇总」当成「原始输出」

**症状**：初稿报告里那张 30 seed 的五方法对照表，数字全部正确——但表格是
**我手写汇总的**，`results/` 里那个所谓「原始文件」也是手写的，不是任何命令
的输出。脚本默认 seed 数是 10，找不到任何 30-seed 日志。

**根因**：不是数字错了，是**溯源链断了**。「数字对」和「数字可核查」是两件事。
手写的表格，你无法回答「如果我怀疑这个数，该跑哪条命令」。

**怎么抓出来的**：`plan → review → execute` 要求新方向提交前交给**独立评审**
逐条挑硬伤。这次评审结论是 REJECT，8 条发现全部成立，第 1 条（SEVERE）
就是它——**评审拦截到的第一次「作者自身诚信瑕疵」**。

**对策**：新增 `final_eval_30.py`，把全部对比策略在 30 seed × 名义/DR
两组下真实重跑，产出 **441 行逐 seed 转录**落盘（含 14 组汇总行）。
此后终版表一律由脚本生成。

> **纪律**：数字的来源必须是**命令输出文件**，不是作者的记忆或汇总。
> 如果一条数字无法回答「跑哪条命令能重现它」，它就不该出现在报告里。

---

## 坑 2：10 个种子得出的结论是错的

**症状**：执行器退化扫描第一轮跑了 10 seed，结论是「ε≤40% 全部 10/10 零碰撞，
50% 档崩 3/10」——一张漂亮的、有明确边界的曲线。

**根因**：换一组全新种子（2000–2009）复跑同一档，**ε=33% 就已经有 1/10 撞车，
50% 档升到 6/10**。10 个种子在这个问题上的置信宽度大到足以改变定性结论。

**对策**：先扩到 **20 seed**（两组各 10 合并）重采全部数据；收尾评审又提出
「20 seed 在 50% 档置信宽度 ±2，发表需补到 50 seed——成本极低，
10 分钟机时换一张无需答辩的图」，于是补到 **50 seed（5 组 × 10）**。
**第二次自我修正比我预想的更值钱**：20 seed 的结论是「ε≤30% 全过、33% 起失效」。
50 seed 显示**失效从 ε=30% 就开始了（3/50）**，而 **ε=30% 对应的 a_max=1.05
仍在域随机化训练分布之内**。即：失效边界不是「训练分布外缘」，
而是**扎在分布内侧**。
| ε | τ (s) | a_max (m/s²) | RL 无碰撞 |
|---|---|---|---|
| 20% | 0.144 | 1.20 | 50/50 |
| **30%** | **0.156** | **1.05** | **47/50** |
| 33% | 0.160 | 1.005 | 41/50 |
| 40% | 0.168 | 0.900 | 37/50 |
| 50% | 0.180 | 0.750 | 27/50 |

> **纪律**：小样本下**不要写定性结论**；第一次跑出漂亮结论时，
> **换组种子再确认一遍**。

---

## 坑 3：单位 bug 让 a_max 变成了负数

**症状**：扫描脚本首轮执行，**全部档位 offtrack**。

**根因**：`eps_list` 里传的是整数百分比（10、20…），但代码把它当分数用：

```
a_max = 1.5 × (1 − 10) = −13.5     # 加速度上限为负
```

加速度上限是负数，小车的物理模型直接被玩坏。
**对策**：`eps = eps_pct / 100.0`，并在 wrapper 构造函数里加
`0 ≤ eps ≤ 0.5` **守卫**。症状（全部 offtrack）和根因（单位错误）
没有逻辑联系——这类 bug 只能靠**在边界处加断言**暴露。

## 坑 4：best_model 反而是一个坏策略

**症状**：训练完成，按回合回报保存的 `best_model.zip` 拿来评估，**超车 0/10 全失败**，
平均车速 **0.23 m/s**。而 `final_model.zip` 是 **10/10 成功、1.4 s**。
0.23 m/s 不是随机数——**它恰好等于领头车的速度**，
即 `best_model` 学到的是 `a ≡ 0` 的**纯跟随策略**。

**根因**：**回合总回报被回合长度偏置了**。跟随局跑满 1500 步、
成功超车局只有约 90 步。绝对回报上「混时长」并不比「早成功」差，
于是按平均回报保存的「最优」模型，其实是最会混时长的那个。
同一偏置在跟车任务也复现了：**final 0.93 cm 优于 best 1.00 cm**。

**对策**：**best 与 final 必须都测**（项目铁律 6）；最终采用 final。
核心认知是——**checkpoint 的选择指标必须与任务目标对齐（成功率 / 间距误差），
而不是回合总回报。**

> **纪律**：训练框架的「best」是按它自己的损失定义的，不是按你的业务指标。
> 你永远要问一句：它说的「好」，是谁定义的好？

## 坑 5：撞车之后，策略变得保守（「恐惧屏障」）

**症状**：超车任务**从零开始训练两次**，累计数千万步交互，
**全部收敛到「永远跟随」**——永不换道。而跟随的回报（≈ +65）
明明低于成功超车（≈ +78）。

**根因**：**值函数学到了「靠近前车 = 危险」**。探索噪声在间距 < 0.15 m 时
导致撞车（−500 惩罚），而它不知道「邻道并行在几何上不可能碰撞」——
横向间距 0.15 m > 碰撞半径 0.12 m。安全超车需要串起
「换道 → 逼近 → 反超」的完整动作链，随机探索串不起来。**回报梯度存在，峡谷太深。**

**对策**：**奖励整形推到尽头也不够用**，正确做法是**改变信息来源**
——规则状态机采 300 局演示（41936 条转移），行为克隆热身，再 PPO 微调：

| 步骤 | 结果 |
|---|---|
| BC 热身（loss 2.0346 → 1.7721） | 2/10 成功、8 次碰撞 |
| PPO 微调 5 M 步 | **10/10 成功、零失误、1.4 s** |

BC 只有 2/10 成功——**这不重要**，它只要「会尝试」就够，时机由 RL 接管后修复。
训练曲线看得很清楚：回合长度先升后降（129 → 142 → 86），先学会不撞车、再学会超车。

> **纪律**：当「发现行为」比「优化行为」更难时，**换信息来源，而不是加奖励权重**。

---

## 坑 6：环境几何不一致，RL 会用「永不换道」告诉你

**症状**：内圈赛道用「外圈向内偏置」的直觉构造后，**超车 0/10，
规则基线也全部冲出赛道**。

**根因**：底边固定在 y=0，导致内圈底边与外圈**完全重合**，顶边间距变成 0.3 m。
四条边间距是 `0 / 0.15 / 0.15 / 0.30`——顶边一换道就冲出，RL 于是
「理性地」学到了永不换道。

**对策**：几何加 y 向偏移，使四边及弧段**全场等距 0.150 m**，
并**数值验证**（对每条外圈点采样到内圈的最小距离，全场 = 0.150 m）。

> **纪律**：**两个完全不同的根因（奖励地形 vs 环境几何）会表现出同一个症状
> （永不换道）**。仅看「策略行为异常」会误诊。所以回归测试必须
> **审计全部终止原因**（碰撞 / 跟丢 / 冲出 / 成功 / 失败）。

---

## 坑 7：依赖 pin 冲突，差点毁掉「环境冻结」存证

**症状**：要装离线 RL 库 d3rlpy，评审在动手前拦了一道。

**根因**：**d3rlpy pin 了 `gymnasium==1.0.0`**，而项目冻结的主环境是
**gymnasium 1.3.0**。直接装进主 venv，1.3.0 会被**悄无声息地降级**——
项目用 5 个核心文件的 sha256 做「环境冻结」存证，**哈希当场作废**。

**对策**：**独立 venv**（`.venv-d3rlpy/`），只读演示数据 `demos_v1.npz`，
绝不碰主环境。装完后 `sha256sum -c` 验证 5 个冻结文件哈希未变；
理由写进 `requirements-offline.txt` 文件头注释，让接手者不会重犯。

> **纪律**：**一个依赖的 pin，可以静默作废一整套复现性存证。**
> 装任何新库之前，先问「它会动我的冻结文件吗」。

---

## 坑 8：两个小的口径坑

- **进度差套圈回绕**：反超后**所有回合都被误判为「跟丢」**。根因是进度差 `δ`
  在 ±L/2 处回绕、而「已反超」标志被复位。对策：**反超标志永不复位**；
  奖励整形也必须在 `|δ| > 1.2` 时禁用，否则回绕处产生伪奖励。
- **评估图互相覆盖**：多组实验的图用了默认文件名，**互相覆盖**。
  对策：`--out` 版本化命名。和坑 4 同类——**默认值是为了方便，不为归档**。---

## 纪律的主线：plan → review → execute

上面 8 个坑里，**坑 1 与坑 7 是流程在动手前/提交前拦下来的，不是事后发现的**。
加上不在这 8 条之列的那些，评审一共拦截了 4 次（下表）：

| 拦截 | 内容 | 出处 |
|---|---|---|
| ① | sb3-contrib **不含任何离线 RL 算法**，库选型错误 | 路线图 §A4 修正 |
| ② | W2 plan 的数据映射描述错误（照原稿写必炸） | `reviews/20260920_w2_plan_review1.md` R1 |
| ③ | d3rlpy 会 pin 降级 gymnasium，必须独立 venv | 同一份评审的「评审评价」段 |
| ④ | 30-seed 数字无命令溯源（坑 1） | `reviews/20260920_report48_review1.md` 发现 #1 |

流程很简单：**任何新方向 / 新实验批次 / 环境改动 / 报告修订之前，
先写一页纸方案（假设 / 方法 / 验收 / 风险），交独立评审逐条挑硬伤，
意见存档 `reviews/`，处理完才动手。** 配套的是**假设预注册**：
判定阈值必须在训练启动**之前**写进假设登记表定死，落在边界就补 seed，
**不放宽阈值**。

这套东西有没有翻过车？有，而且我把它写进了登记表：

> H4 的判定阈值在数据效率扫描**已部分出数之后**才被改写。
> 按预注册制度的严格标准，**这是 post-hoc 修订，不满足「训练前定死」**。
> 如实记录：原判据判 H4 **证伪**，新判据判 **成立**（差异源于「BC」指代
> 哪个实现），**故 H4 只是「条件性成立」。**

反面教材还有一条：评审发现我写过「H1–H3 阈值训练前登记」的过度声明，
而后来补的 H6 恰好**落在阈值边界**（max−min = 2/10，阈值也是 2/10），
按自定规则应当补 seed——这条当时**没补**，如实记为遗留；
后来补到 30 seed（max−min = 5/30，仍 ≤ 2/10）才闭合。**为什么这套流程值钱：评审者没有执行权。** 作者无法为自己开脱，
评审结论只能「接受」或「接受并说明」——所以它才能在坑 1 里，
拦下**作者自己写的那张看起来很漂亮的表**。

## 附：三条可以立刻拿走的东西

1. **开训前先算更新次数**：`总步数 / (环境数 × n_steps)`。
   第一版 50 万步、`32 × 1024` 配置只有约 15 次梯度更新，策略行为和随机无异；
   PPO 通常需要 300 次以上。本项目三笔账是 305 / 610 / 610。
2. **审计全部终止原因**：五类全报——只报成功率会掩盖「假超车」这类假象。
3. **数字必须来自命令输出**。这条排第一不是因为它最重要，
   而是因为它**最容易被自己放过**。

---

## English Abstract

We report the engineering discipline that emerged from 14 recorded
failure-diagnosis-fix rounds while building an RL-based following and overtaking
system for low-speed AGV fleets. Eight pitfalls are documented with symptom, root
cause, and remedy, drawn from real incidents: a checkpoint selected by mean episode
return that turned out to be a pure follow-forever policy (0/10, mean speed exactly
equal to the leader's), a "fear barrier" that made pure RL converge to never changing
lanes across tens of millions of interaction steps, a unit bug that drove the
acceleration limit negative, a 10-seed sweep whose qualitative conclusion reversed at
20 and again at 50 seeds, and a dependency pin (d3rlpy → gymnasium 1.0.0) that would
have silently downgraded the frozen environment and invalidated the sha256
reproducibility records. Underlying all of them is a plan → review → execute protocol
with pre-registered numeric thresholds, whose review step — deliberately given no
execution authority — intercepted four real defects before they reached execution,
including one instance of the authors' own data-provenance lapse (a hand-written
summary table presented as raw output). We also disclose a post-hoc threshold revision
that violates our own pre-registration rule.

---

## 数字溯源表

> 规则同前：每个数字对应仓库中真实文件与具体位置。

### 关于「14 轮」

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 14（表 3 行数） | `project_paper/project_report_full.md` | 第 591 行标题「**表 3　十四轮"失败—诊断—修复"迭代**」，表体第 595–608 行（14 行，编号 #1–#14） |
| 同表在方法论文中的版本 | `project_paper/project_paper.md` | 第 405 行「**表 3　十四轮**」，表体 14 行 |
| RUNLOG 共 22 条 | `RUNLOG.md` | 16 条 `^### \[2026-09-19\] R` 条目 + 6 条 `^## \[2026-09-20\]` 阶段条目 |
| 表 3 多数事件早于 RUNLOG | 推理自 `RUNLOG.md` 首个条目为 `R0.1 跟车健康检查`（Phase 0），而表 3 含「NumPy 2.x 数组转标量」「固定预瞄切弯」等更早事件 | 无原始日志（**已在正文如实说明**） |

### 坑 1（溯源缺失）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 评审 REJECT、8 条发现、#1 为 SEVERE | `reviews/20260920_report48_review1.md` | 「评审结论」+ 发现 #1 表行 |
| `final_table_30seed.txt` 为手写汇总 | 同上 | 发现 #1 描述「是手写汇总，非命令输出；脚本默认 10 seed，无 30-seed 日志」 |
| `final_eval_30.py` 产出 441 行转录 | `results/20260920_rq2_offline/final_table_30seed_transcript.txt` | `wc -l` = 441；文件头 `# generated by final_eval_30.py` |
| 「首次拦截到作者自身诚信瑕疵」 | `reviews/20260920_report48_review1.md` | 「流程反思」段引文 |

### 坑 2（种子数）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 首轮 10 seed：ε≤40% 全 10/10、50% 崩 3/10 | `RUNLOG.md` | `[2026-09-20] Phase A W1 — 鲁棒性连续扫描` →「执行中发现并修复的两个问题」第 2 条 |
| 换种子后 ε=33% 有 1/10、50% 升到 6/10 | 同上 | 同段原文 |
| 改用 20 seed（两组各 10 合并） | 同上 | 「→ 10 seed 结论不可靠。**改用 20 seed（两组各 10 合并）**，重采全部数据」 |
| 20-seed 表（ε=30% 20/20、33% 19/20、50% 11/20） | 同上 | 「最终结果（20 seed 协议，固定 1000–1009 + 2000–2009）」代码块 |
| 评审要求补到 50 seed、≈10 分钟机时 | `RUNLOG.md` | `[2026-09-20] Phase A W1 补充 — 50-seed 复跑（评审意见 2）` →「动机」 |
| 50 seed = 5 组 × 10 | 同上 | 命令 `--seeds 1000..1009 2000..2009 3000..3009 4000..4009 5000..5009` |
| 20→50 seed 结论修正：失效从 ε=30% 起（3/50） | 同上 | 「**对 20-seed 结论的修正（重要）**」段 |
| 表（ε=20/30/33/40/50 的 τ、a_max、无碰撞） | `results/20260926_robustness_sweep/robustness_sweep.csv` | `policy=RL(PPO)` 各 `eps_pct` 行的 `tau` / `a_max` / `collision_free` / `n_seeds` 列 |
| 同一表格的报告版 | `project_paper/project_report_full.md` | §5.4.1 表 5.5 |

### 坑 3（单位 bug）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| `a_max = 1.5×(1−10) = −13.5`、全部 offtrack | `RUNLOG.md` | `Phase A W1` →「执行中发现并修复」第 1 条（原文：`a_max = 1.5×(1−10) = −13.5`（负值）→ 全部 offtrack） |
| 修复 `eps = eps_pct/100` + `assert 0≤eps≤0.5` | 同上 + `robustness_sweep.py` | RUNLOG 同段；代码第 41–46 行 `__init__(self, eps)` 守卫（`if not (0.0 <= eps <= 0.5): raise ValueError`，docstring 明写 "A percent value (e.g. 10) is a units bug"）、第 149 行 `eps = eps_pct / 100.0` |

### 坑 4（checkpoint 偏置）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| best 0/10、`mean_v=0.23`、全部 `failed`、`steps=1501` | `results/20260919_phase2_overtake/eval_best_ot_verbose.txt` | 汇总行 `RL(PPO) overtake=0/10 ... mean_v=0.23 m/s` + 逐 seed 明细 `steps=1501 failed` |
| final 10/10、1.4 s | `results/20260919_phase2_overtake/eval_final_nominal.txt` | 汇总行 |
| 0.23 m/s = 领头车速度 = zero-action 跟随 | `RUNLOG.md` | `R2.6` →「`mean_v=0.23 m/s` = 领头车速度 = zero-action 跟随策略」 |
| 跟车 final 0.93 < best 1.00 | `results/20260919_phase1_follow/eval_final_nominal.txt` / `eval_best_nominal.txt` | 各自 `RL(PPO) mean\|e\|=` 行 |
| 「回合回报被回合长度偏置」跨任务普遍性 | `project_paper/project_report_full.md` | §5.7 + §10.2 末段 |
| 铁律 6（best/final 双测） | `DATA_MANAGEMENT.md` | §3 末「**best/final 双测**（铁律 6）」 |

### 坑 5（恐惧屏障）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 从零探索数千万步全收敛到跟随 | `project_paper/project_report_full.md` | §4.5「本文多次从零探索训练、累计数千万步交互均收敛到跟随策略」 |
| 执行两次 | `project_paper/project_report_full.md` | §6.2「超车任务中 RL 两次收敛到"永不换道"的跟随策略」 |
| 跟随 ≈+65 < 成功 ≈+78 | `project_paper/project_report_full.md` | §6.2；另 §4.5 作「≈+75 以上」，§5.3 表 2 零动作行 |
| 横向 0.15 m > 碰撞半径 0.12 m | `overtake_env.py` | 第 44 行车道偏置 0.15；第 28 行 `COLLISION_2D = 0.12` |
| BC loss 2.0346 → 1.7721 | `results/20260919_phase2_overtake/pretrain.log` | `BC epoch 1/10 loss=2.0346` … `BC epoch 10/10 loss=1.7721` |
| 演示 300 局 / 41936 转移 | 同上 | `demo dataset: 41936 transitions` |
| BC 2/10、8 碰撞 | `results/20260919_phase2_overtake/eval_bc.txt` | 汇总行 `RL(PPO) overtake= 2/10 collision=8` |
| 微调后 10/10、1.4 s | `eval_final_nominal.txt` | 汇总行 |
| ep_len 129 → 142 → 86 | `results/20260919_phase2_overtake/train_finetune.log` | 首行 `ep_len_mean 129`、末行 `86.5`；峰值 142 见 `RUNLOG.md` R2.3「parsed 77 points ... max=142」 |
| 奖励整形推到尽头仍不足 | `project_paper/project_report_full.md` | §6.2 末「reward shaping（4.4 节）推到尽头仍不足，最终以 LfD 管线（4.5 节）解决」 |

### 坑 6（环境几何）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 超车 0/10 且规则基线全冲出 | `project_paper/project_report_full.md` | 表 3 第 5 行（第 599 行） |
| 底边重合、顶边间距 0.3 m | `project_paper/project_report_full.md` | §6.3「实际四边间距为 0/0.15/0.15/0.30 m」 |
| 修复后全场 = 0.150 m | `project_paper/project_report_full.md` | §6.3「全场 = 0.150 m」；代码 `overtake_env.py` 第 44 行 `y0=0.15` |
| 五类终止原因全审计 | `DATA_MANAGEMENT.md` | §1 铁律 5 / `AGENT_HANDOFF.md` §5 铁律 5 |

### 坑 7（依赖 pin）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| d3rlpy pin `gymnasium==1.0.0`，主环境 1.3.0 | `requirements-offline.txt` | 文件头注释「d3rlpy pins gymnasium==1.0.0, which would DOWNGRADE the frozen main environment's gymnasium 1.3.0 and invalidate the sha256 freeze records」 |
| 评审「动手前拦截级」评价 | `reviews/20260920_w2_plan_review1.md` | 「评审评价」段 |
| 独立 venv `.venv-d3rlpy/` | `AGENT_HANDOFF.md` | §1「两个 venv（**不要合并**）」 |
| 5 个冻结文件 sha256 保持不变 | `RUNLOG.md` | `Phase A W2` →「环境隔离验证：... 5 个冻结文件 sha256 **全部 OK**」 |
| 5 个哈希值 | `DATA_MANAGEMENT.md` | §8 哈希表 |

### 坑 8（小坑）

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 进度差 ±L/2 回绕、反超标志永不复位 | `project_paper/project_report_full.md` | 表 3 第 7 行（第 601 行）；代码 `overtake_env.py` 第 229–231 行注释 `NOTE: ahead_flag is NEVER reset - delta wraps at +-L/2` |
| 整形在 `\|δ\|>1.2` 时禁用 | `overtake_env.py` | 第 222–225 行 `# Disabled near the +-L/2 wrap (\|delta\| > 1.2)` + `if abs(delta) < 1.2 and abs(self.prev_delta) < 1.2:` |
| 评估图互相覆盖 | `project_paper/project_report_full.md` | 表 3 第 14 行（第 608 行）；`DATA_MANAGEMENT.md` §4 |

### 流程部分

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 4 次拦截的**具名四条** | ①`research/roadmap.md` §A4 行（sb3-contrib 注释）+ commit `1e6b786`；②`reviews/20260920_w2_plan_review1.md` R1；③`requirements-offline.txt` 头注释 + 同评审「评审评价」段；④`reviews/20260920_report48_review1.md` 发现 #1 | 见「4 次拦截」表各行 |
| 「评审流程已拦截 4 次真问题」原文 | `AGENT_HANDOFF.md` | §5 末句 |
| plan → review → execute | `DATA_MANAGEMENT.md` §10 | （**注：本文件现存版本仅到 §9；§10 在引用它的各处文档中被指称，正文按「被指称的流程」表述**） |
| H4 为 post-hoc 修订 | `research/ASSUMPTIONS.md` | H4「修订说明（诚信披露）」段 |
| H6 阈值边界 max−min = 2/10 | `research/ASSUMPTIONS.md` | H6 检验结果表 10-seed 行 |
| H6 补 30 seed 后 max−min = 5/30 | `research/ASSUMPTIONS.md`；`results/20260920_rq2_offline/h6_boundary_30seed.txt` | 登记表 30-seed 行；文件末 `H6 verdict (30-seed): max=5/30 min=0/30 spread=5/30` |
| 未闭合项「H6 边界未补 seed」如实记为遗留 | `reviews/20260920_report48_review1.md` | 「未闭合项（如实遗留）」段 |
| 「H1–H3 阈值训练前登记」过度声明被撤销 | 同上 | 发现 #2 表行 |
| 评审者无执行权 | `reviews/20260920_report48_review1.md` | 「流程反思」段「证明"评审者无执行权"的设计有效（作者无法为自己开脱）」 |

### 三条速取

| 数字 | 出处文件 | 位置 |
|---|---|---|
| 15 次更新（50 万步 / 32×1024） | `project_paper/project_report_full.md` | §6.1 `5×10^5/(32×1024) ≈ 15`；表 3 第 1 行 |
| PPO 需 300 次以上更新 | `project_paper/project_report_full.md` | §6.1「而 PPO 通常需要 300 次以上更新」 |
| 305 / 610 / 610 | `results/20260919_phase1_follow/config.json`、`results/20260919_phase2_overtake/config.json` | `updates_stage1` / `updates_stage2` / `updates`；报告 §9.4 表 |
| 五类终止原因 | `AGENT_HANDOFF.md` §5 铁律 5；`project_paper/project_report_full.md` §4.8.1 | 同上 |
| 「假超车」案例 | `project_paper/project_report_full.md` | 表 3 第 6 行（第 600 行）「规则"超车成功"是假象」+ §6.3「仅看成功率会掩盖假象（案例 6）」 |

---

### 存疑 / 需评议（重点核对这三条）

1. **「14 轮」的计数对象**（同草稿 1 的存疑 2）。标题数字取 `表 3` 的 14 行，
   但 `RUNLOG.md` 是 22 条。**两者不是同一计数对象**，且表 3 的多数事件
   在仓库内无逐条原始日志。我选择保留「14」并在正文首节显式说明口径——
   若要求「标题数字必须能在 RUNLOG 里逐条对上」，**这个标题就得改**。
2. **「4 次拦截」的数目与清单自相矛盾**。`AGENT_HANDOFF.md` §5 末称 4 次，
   但括号内只列了 3 项；评审记录自身的序号也彼此对不齐
   （W2 review 自称「第二次」、report48 review 自称「第四次」）。
   正文我用了**具名四条的版本**（①sb3-contrib ②W2 plan 数据映射
   ③d3rlpy venv ④30-seed 溯源），**与 AGENT_HANDOFF 的「4」数目一致、
   但清单不同**。这一条建议对外前统一口径。
3. **`DATA_MANAGEMENT.md` §10 是否存在**。多份文档引用「`DATA_MANAGEMENT.md` §10
   （plan → review → execute）」作为流程依据，但我实际读取的
   `DATA_MANAGEMENT.md` **只有一个 §9（数据资产哈希）作为末节，没有 §10**。
   流程本身真实存在（有 3 份 `reviews/` 记录 + 假设登记表 + commit `7670b4c`
   「docs: codify plan-review-execute process」），但**引用指向的章节号
   在当前文件中找不到**。正文我改用了「被指称的流程」的措辞，
   建议核实是否需要补写该章节或以其他文件为准。

> 草稿版本：v1（未发布）。项目仓库：`/home/zy/car_rl/code0919`。
