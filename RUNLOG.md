# RUNLOG — 复刻实验记录

> 规范：**每一次训练或评估运行记一条**（日期 / 运行 ID / commit / 命令 / 超参 /
> **原始输出** / 验收结论 / 与论文基准的差异 / 产物路径）。环境或超参一变即新增
> 条目，旧条目**只增不改**（铁律 7）。
> 铁律：①数据真实性 ②环境冻结 ③健康检查先行 ④更新次数≥300 ⑤终止原因全审计
> ⑥checkpoint best/final 双测、图版本化 ⑦每轮记录、旧文件归档不删。
>
> 运行 ID 约定：`R<阶段>.<序号>`；训练记 `[train]`，评估记 `[eval]`。

---

## Phase 0 — 环境初始化与健康检查

### [2026-09-19] R0.1 `[eval]` 跟车健康检查（car_following_sim）

- **命令**: `uv run python car_following_sim.py`
- **超参**: 脚本默认（T=45 s, v_set=0.3 m/s, d_des=0.2 m, 纯 P 跟随）
- **原始输出**:

  ```
  == sim done: T=45.0s  v_set=0.3m/s  d_des=0.2m ==
    mean|gap err| = 1.10 cm
    max |gap err| = 2.50 cm
    last gap      = 18.78 cm
    follower lateral RMS = 2.66 cm
    near-collision steps (gap<12cm) = 0
  ```

- **验收结论**: mean|e| = 1.10 cm ≤ 1.2 cm ✅，零碰撞 ✅
- **与论文基准差异**: 此脚本为 Week1 纯 P 演示（非表 1 的"规则 P 撞车"口径），
  仅作健康检查用，不与表 1 直接对照。
- **产物**: `results/20260919_phase0_healthcheck/fig_follow_baseline_sim.png`

### [2026-09-19] R0.2 `[eval]` 超车规则基线健康检查（rule-only）

- **命令**: `uv run python train_ot.py eval --rule-only`
- **超参**: 固定 10 seed（2000~2009），domain_randomize=False，仅规则状态机
- **原始输出**:

  ```
  evaluating 10 fixed seeds (domain_randomize=False)
  rule-based   overtake=10/10  collision=0  offtrack=0  lost=0  t_overtake=  2.0s  mean_v=0.55 m/s
  ```

- **验收结论**: overtake=10/10、collision=0、offtrack=0、lost=0 ✅（铁律 3 达标）
- **与论文基准差异**: 2.0 s / 0.55 m/s 与论文表 2 完全一致。
- **产物**: `results/20260919_phase0_healthcheck/{eval_overtake_ruleonly.txt, .png}`

---

## Phase 1 — 跟车任务复刻

> 共用超参：PPO MlpPolicy[128,128], lr 3e-4, n_steps 256, batch 512, n_envs 32,
> γ 0.99, GAE λ 0.95, clip 0.2, ent_coef 0.01, seed 0。

### [2026-09-19] R1.1 `[train]` 跟车 stage1（easy 课程）

- **commit**: 3f2dfdc
- **命令**: `uv run python train_ppo.py train --easy --timesteps 2500000 --n-envs 32 --seed 0`
- **更新次数（铁律 4）**: 2.5M/(32×256) = **305** ✅
- **原始输出（曲线特征，make_curves.py 解析）**:

  ```
  $ uv run python make_curves.py results/20260919_phase1_follow/train_stage1.log ...
  parsed 306 points
    final ep_rew_mean = -88.0
    ep_len_mean: first=145 last=1000 min=145 max=1000
  ```
  末段收敛：ep_rew_mean 稳定段（末 10%）≈ −86；早期瞬态最低 −5000(@0.03 M)。
  训练耗时 time_elapsed = 720 s。

- **验收结论**: ep_len 升至满回合 1000（全程不碰撞），课程目标达成 ✅
- **与论文基准差异**: 命令步数由论文的 2,000,000 上调至 2,500,000（论文原值仅
  244 次更新，违反铁律 4），详见报告 9.4。
- **产物**: `results/20260919_phase1_follow/{train_stage1.log, fig_follow_curve_stage1.png}`；
  checkpoint `ckpt/follow_stage1_v1.zip`（best）、`ckpt/follow_stage1_final_v1.zip`

### [2026-09-19] R1.2 `[train]` 跟车 stage2（完整任务，从 stage1 best 续训）

- **commit**: 3f2dfdc
- **命令**: `uv run python train_ppo.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt/best_model.zip`
- **更新次数（铁律 4）**: 5M/(32×256) = **610** ✅
- **原始输出（曲线特征）**:

  ```
  parsed 611 points
    final ep_rew_mean = -910.0
    ep_len_mean: first=94 last=974 min=94 max=1000
  ```
  末段收敛：ep_rew_mean 稳定段（末 10%）≈ −864；起始 ep_len 94（课程切换后
  初期撞车）→ 稳定段 989。训练耗时 time_elapsed = 1605 s。

- **验收结论**: 完整任务下策略收敛，ep_len 由 94 回升至 974 ✅
- **与论文基准差异**: 无（超参、步数均同论文）。
- **产物**: `results/20260919_phase1_follow/{train_stage2.log, fig_follow_curve_stage2.png}`；
  checkpoint `ckpt/follow_stage2_best_v1.zip`、`ckpt/follow_stage2_final_v1.zip`

### [2026-09-19] R1.3 `[eval]` 跟车 best 模型（名义参数）

- **命令**: `uv run python train_ppo.py eval --model ckpt/best_model.zip -v --out .../fig_follow_eval_best_nominal.png`
- **超参**: 固定 10 seed（1000~1009），domain_randomize=False
- **原始输出**:

  ```
  RL(PPO)      mean|e|=  1.00 cm  worst|e|=150.74 cm  collision=0 lost=0 offtrack=0 /10
  rule P       mean|e|= 37.98 cm  worst|e|=150.74 cm  collision=10 lost=0 offtrack=0 /10
  rule P+FF    mean|e|=  1.93 cm  worst|e|=150.74 cm  collision=0 lost=0 offtrack=0 /10
  ```

- **验收结论**: RL 1.00 cm ≤ 1.3 cm、零碰撞 ✅；P 复现 10/10 撞车 ✅；P+FF 1.93 cm ✅
- **与论文基准差异**: 与表 1 逐字一致（RL 1.00、P 37.98/撞车、P+FF 1.93）。
- **终止原因审计（铁律 5）**: RL 与 P+FF 全部 timeout（跑满 1001 步）；P 全部
  collision（平均存活 ~52 步）。
- **产物**: `results/20260919_phase1_follow/{eval_best_nominal.txt, fig_follow_eval_best_nominal.png}`

### [2026-09-19] R1.4 `[eval]` 跟车 best 模型（域随机化）

- **命令**: `uv run python train_ppo.py eval --model ckpt/best_model.zip --domain-randomize -v --out .../fig_follow_eval_best_dr.png`
- **超参**: 固定 10 seed（1000~1009），domain_randomize=True
- **原始输出**:

  ```
  RL(PPO)      mean|e|=  1.06 cm  worst|e|=172.89 cm  collision=0 lost=0 offtrack=0 /10
  rule P       mean|e|= 43.55 cm  worst|e|=172.89 cm  collision=10 lost=0 offtrack=0 /10
  rule P+FF    mean|e|=  1.94 cm  worst|e|=172.89 cm  collision=0 lost=0 offtrack=0 /10
  ```

- **验收结论**: RL 1.06 cm（较名义退化 6%）、零碰撞 ✅
- **与论文基准差异**: 与表 1 逐字一致（RL 1.06、P 43.55/撞车、P+FF 1.94）。
- **产物**: `results/20260919_phase1_follow/{eval_best_dr.txt, fig_follow_eval_best_dr.png}`

### [2026-09-19] R1.5 `[eval]` 跟车 final 模型（名义参数，铁律 6 双测）

- **命令**: `uv run python train_ppo.py eval --model ckpt/final_model.zip -v --out .../fig_follow_eval_final_nominal.png`
- **超参**: 固定 10 seed（1000~1009），domain_randomize=False
- **原始输出**:

  ```
  RL(PPO)      mean|e|=  0.93 cm  worst|e|=150.74 cm  collision=0 lost=0 offtrack=0 /10
  rule P       mean|e|= 37.98 cm  worst|e|=150.74 cm  collision=10 lost=0 offtrack=0 /10
  rule P+FF    mean|e|=  1.93 cm  worst|e|=150.74 cm  collision=0 lost=0 offtrack=0 /10
  ```
  逐 seed（final）：0.57/0.73/0.57/0.71/0.73/0.79/0.86/0.98/1.49/1.82 cm。

- **验收结论**: RL 0.93 cm，逐 seed 全胜 P+FF ✅
- **与论文基准差异**: **final(0.93) 优于 best(1.00)**——复现 5.7 节 checkpoint
  偏置的跨任务普遍性（报告 10.2 末）。
- **产物**: `results/20260919_phase1_follow/{eval_final_nominal.txt, fig_follow_eval_final_nominal.png}`

### [2026-09-19] R1.6 `[eval]` 跟车 final 模型（域随机化，铁律 6 双测）

- **命令**: `uv run python train_ppo.py eval --model ckpt/final_model.zip --domain-randomize --out .../fig_follow_eval_final_dr.png`
- **超参**: 固定 10 seed（1000~1009），domain_randomize=True
- **原始输出**:

  ```
  RL(PPO)      mean|e|=  0.91 cm  worst|e|=172.89 cm  collision=0 lost=0 offtrack=0 /10
  ```

- **验收结论**: RL 0.91 cm、零碰撞 ✅（域随机化下 final 甚至略优于名义）
- **与论文基准差异**: 优于论文 best 口径 1.06 cm。
- **产物**: `results/20260919_phase1_follow/{eval_final_dr.txt, fig_follow_eval_final_dr.png}`

**Phase 1 小结**: 6 次运行全部通过；RL 1.00/1.06 cm（best）、0.93/0.91 cm（final），
P 10/10 撞车、P+FF 1.93/1.94 cm——与表 1 逐项一致。checkpoint
`ckpt/follow_stage{1,2}_{v1,best_v1,final_v1}.zip`。

---

## Phase 2 — 超车任务复刻（LfD 管线）

> 共用超参同 Phase 1。

### [2026-09-19] R2.1 `[train]` BC 行为克隆热身

- **commit**: 6503a13
- **命令**: `uv run python train_ot.py pretrain --n-demos 300 --bc-epochs 10`
- **原始输出**:

  ```
  collecting 300 rule-based demonstrations ...
  demo dataset: 41936 transitions
    BC epoch 1/10  loss=2.0346
    BC epoch 2/10  loss=1.9501
    ...
    BC epoch 10/10  loss=1.7721
  BC model -> ckpt_ot/bc_model.zip
  ```

- **验收结论**: 数据集 41936 条（论文"4.2 万" ✅），loss 单调下降 ✅
- **与论文基准差异**: 无。
- **产物**: `results/20260919_phase2_overtake/pretrain.log`；checkpoint `ckpt_ot/overtake_bc_v1.zip`

### [2026-09-19] R2.2 `[eval]` BC 策略评估

- **命令**: `uv run python train_ot.py eval --model ckpt_ot/bc_model.zip -v`
- **超参**: 固定 10 seed（2000~2009），domain_randomize=False
- **原始输出**:

  ```
  rule-based   overtake=10/10  collision=0   offtrack=0  lost=0  t_overtake= 2.0s  mean_v=0.55 m/s
  RL(PPO)      overtake= 2/10  collision=8   offtrack=0  lost=0  t_overtake= 2.2s  mean_v=0.50 m/s
  ```

- **验收结论**: BC 2/10 成功、8 碰撞——**符合论文 4.5 节预期**（BC 精度无需完美）✅
- **与论文基准差异**: 与论文 4.5 节"仅 2/10 成功，8 次碰撞"完全一致。
- **产物**: `results/20260919_phase2_overtake/{eval_bc.txt, fig_overtake_bc_eval.png}`

### [2026-09-19] R2.3 `[train]` 超车 PPO 微调（从 bc_model 续训）

- **commit**: 6503a13
- **命令**: `uv run python train_ot.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt_ot/bc_model.zip`
- **更新次数（铁律 4）**: 5M/(32×256) = **610** ✅
- **原始输出（曲线特征）**:

  ```
  parsed 77 points
    final ep_rew_mean = 72.5
    ep_len_mean: first=129 last=86 min=85 max=142
  ```
  "先升后降"：ep_len 129 → 142(@0.39 M) → 86；ep_rew −277 → +72.5。
  训练耗时 time_elapsed = 7398 s。

- **验收结论**: 曲线形状复现论文 5.5 节（129→140→86），回报收敛 ≈+72 ✅
- **与论文基准差异**: 峰值 142 vs 论文 140（+2 步，随机波动内）。
- **产物**: `results/20260919_phase2_overtake/{train_finetune.log, fig_overtake_curve_finetune.png}`；
  checkpoint `ckpt_ot/overtake_best_v1.zip`、`ckpt_ot/overtake_final_v1.zip`

### [2026-09-19] R2.4 `[eval]` 超车 final 模型（名义参数）

- **命令**: `uv run python train_ot.py eval --model ckpt_ot/final_model.zip -v --out .../fig_overtake_final_nominal.png`
- **超参**: 固定 10 seed（2000~2009），domain_randomize=False
- **原始输出**:

  ```
  rule-based   overtake=10/10  collision=0  offtrack=0  lost=0  t_overtake=  2.0s  mean_v=0.55 m/s
  RL(PPO)      overtake=10/10  collision=0  offtrack=0  lost=0  t_overtake=  1.4s  mean_v=0.71 m/s
  ```
  逐 seed 超车耗时：1.1/1.2/1.3/1.3/1.4/1.4/1.4/1.5/1.6/1.6 s，全部 success。

- **验收结论**: 10/10 success、零失误、t_ot=1.4 s ≤ 2.0 s ✅
- **与论文基准差异**: 与表 2 逐字一致（RL 10/10、1.4 s、0.71 m/s；规则 2.0 s、0.55 m/s）。
- **终止原因审计（铁律 5）**: 全部 success，collision/offtrack/lost/failed 均为 0。
- **产物**: `results/20260919_phase2_overtake/{eval_final_nominal.txt, fig_overtake_final_nominal.png}`

### [2026-09-19] R2.5 `[eval]` 超车 final 模型（域随机化）

- **命令**: `uv run python train_ot.py eval --model ckpt_ot/final_model.zip --domain-randomize -v --out .../fig_overtake_final_dr.png`
- **超参**: 固定 10 seed（2000~2009），domain_randomize=True
- **原始输出**:

  ```
  rule-based   overtake=10/10  collision=0  offtrack=0  lost=0  t_overtake=  2.0s  mean_v=0.54 m/s
  RL(PPO)      overtake=10/10  collision=0  offtrack=0  lost=0  t_overtake=  1.4s  mean_v=0.70 m/s
  ```

- **验收结论**: 10/10 success、零失误、1.4 s——**与名义完全一致，零退化** ✅
- **与论文基准差异**: 复现论文 5.4 节"性能完全不退化"。
- **产物**: `results/20260919_phase2_overtake/{eval_final_dr.txt, fig_overtake_final_dr.png}`

### [2026-09-19] R2.6 `[eval]` 超车 best 模型（名义参数，铁律 6 双测）

- **命令**: `uv run python train_ot.py eval --model ckpt_ot/best_model.zip -v --out .../fig_overtake_best_nominal.png`
- **超参**: 固定 10 seed（2000~2009），domain_randomize=False
- **原始输出**:

  ```
  RL(PPO)      overtake=0/10  collision=0  offtrack=0  lost=0  t_overtake= nans  mean_v=0.23 m/s
    逐 seed 明细：全部 reason=failed, overtaken=False, steps=1501（跑满超时）
    mean_v=0.23 m/s = 领头车速度 = zero-action 跟随策略
  ```

- **验收结论**: best = 0/10 跟随策略（**任务失败**），不可采用 ✅（双测价值体现）
- **与论文基准差异**: **复现论文 5.7 节 checkpoint 偏置**——best 是按回合总回报
  选的，被回合长度偏置，反而不是任务最优。故 final（R2.4，10/10）为最终采用模型。
- **产物**: `results/20260919_phase2_overtake/{eval_best_ot.txt, eval_best_ot_verbose.txt, fig_overtake_best_nominal.png}`

**Phase 2 小结**: 6 次运行全部通过；final 10/10、1.4 s、零失误（名义与 DR 一致）；
best 0/10 跟随（复现 5.7 节）；BC 2/10（复现 4.5 节）。checkpoint
`ckpt_ot/overtake_{bc,best,final}_v1.zip`。

---

## Phase 3 — 数据与版本管理规范化

### [2026-09-19] R3.1 `[mgmt]` 建立并执行管理规范

- **commit**: 897a07e
- **执行内容**:
  1. `.gitignore` 排除 `.venv/`、`__pycache__/`、`tb_logs*/`、`*.zip`、`*.npz`、
     根目录 `*.png`、`*:Zone.Identifier`。
  2. git tag 里程碑：`milestone/phase0-env`(3f2dfdc)、`milestone/phase1-follow`(6503a13)、
     `milestone/phase2-overtake`(897a07e)；模型用 tag + 目录归档，不入版本库。
  3. `results/YYYYMMDD_<实验名>/` 规范：config.json + eval_*.txt + train_*.log + fig_*.png。
  4. checkpoint 命名 `<任务>_<阶段>_v<N>.zip`，禁止裸名覆盖。
  5. 新增 `DATA_MANAGEMENT.md`；追加 5 个代码文件 sha256 哈希表（环境冻结存证）。
- **验收结论**: 规范建立并落地，Phase 1/2 产物全部归档 ✅
- **与论文基准差异**: N/A
- **产物**: `DATA_MANAGEMENT.md`、`results/*/config.json`、各 tag

---

## Phase 4 — 撰写完整项目报告

### [2026-09-19] R4.1 `[doc]` 报告撰写与修订

- **commit**: fb6b684（初版）+ 本次修订
- **执行内容**:
  1. 以 `project_paper.md` 为基底扩写为 `project_report_full.md`（保留 1~8 章与
     全部公式），新增第 9~12 章"复刻验证"、附录 C"数据与版本管理"、图索引表。
  2. **报告修订（本轮）**：统一 10.1 表格与图 10 口径（首点/早期极值/稳定段）；
     清除全部"请插入"占位符、插图清单路径统一 `figures_v2/`；11.1② 插图 13、
     11.4 插图 12；附录 A stage1 命令改 2,500,000 并加脚注引用 9.4；10.2 末补
     final 优于 best 的 checkpoint 偏置普遍性。
- **验收结论**: 报告完成并修订，数字全部可溯源（附录 C.5）✅
- **与论文基准差异**: 如实记录唯一偏差（stage1 步数，报告 9.4）。
- **产物**: `project_paper/project_report_full.md`、`project_paper/figures_v2/`（13 图）

---

## [2026-09-20] Phase A 前置 — 演示数据集导出（RQ2 输入）

- **commit**: （本批）
- **任务**: 路线图 §4 立即动作② —— export_demos.py 导出 BC 演示数据集并算 sha256
- **命令与原始输出**:

  ```
  $ uv run python export_demos.py --n-demos 300 --out demos_v1.npz
  collecting 300 rule-based demos (seed base 30000) ...
  dataset: 41936 transitions, 300 episodes
    obs dim = 6, act dim = 2
    rewards: mean=0.549 min=-0.1 max=60.3
  episode outcomes: success 300 (100%)
  wrote demos_v1.npz (1.38 MB)
  sha256 4b31c47e7605df0eb269a02eeb46d8f6dbab97cf7b312e38d49045193c47f429  demos_v1.npz
  ```

- **数据完整性校验**: next_obs[t] == obs[t+1] 逐点吻合；300 集各含真终止标记；
  观测 6 维 / 动作 2 维与环境一致。
- **验收结论**: 41936 转移 = 论文「4.2 万」一致；seed 族与 pretrain 一致（可与 BC 2/10 对比）。
- **与论文基准差异**: 300/300 success（规则机在演示 seed 下从不失败，与 rule-only 10/10 一致）；
  数据集为 **positive-only**（无失败轨迹）——已记录于 demos_v1_README.md 的「用途与边界」。
- **环境冻结**: export_demos.py 仅依赖 numpy，未装 d3rlpy（其 pin gymnasium==1.0.0，
  会降级本项目 1.3.0）——d3rlpy 留到 Phase A W2 独立 venv 加载。
- **gitignore 例外**: 加 `!demos_v1.npz`（否则 `*.npz` 静默吞掉发布资产，评审已预警）。
- **产物**: demos_v1.npz、demos_v1_README.md、export_demos.py、DATA_MANAGEMENT §9
- **遗留问题**: 无
- **下一步**: Phase A W1（鲁棒性扫描 A1/A2 + SAC 消融 A3）

---

## [2026-09-20] Phase A W1 — 鲁棒性连续扫描（A1/A2）

- **commit**: （本批）；plan: `plan_A1_A2.md`；review: `reviews/20260920_robustness_plan_review1.md`
- **流程**: DATA_MANAGEMENT §10（plan → review → execute）首次正式落地
- **命令**:

  ```
  uv run python robustness_sweep.py --only-0          # R3 试点门
  uv run python robustness_sweep.py --eps-list 0 10 20 30 33 40 50
  ```

- **方法**: `RobustnessWrapper` 在 `reset()` 后只改写 `follower.tau/a_max`
  （τ↑、a_max↓ 退化），leader 保持名义（R1）；冻结环境零改动（铁律 2）。
- **R3 试点门（先于全档）**: ε=0 复现归档名义结果 **RL 0.93 / P+FF 1.93 cm**
  ——wrapper 无侵入性，且证明 leader 未被误改 ✅

- **执行中发现并修复的两个问题（均属流程价值）**:
  1. **单位 bug**：首轮 main() 误把 `eps_list` 的整数（10/20…）当分数传入，
     `a_max = 1.5×(1−10) = −13.5`（负值）→ 全部 offtrack。修复：`eps = eps_pct/100`，
     并在 wrapper 构造加 `assert 0≤eps≤0.5` 守卫（防再犯）。
  2. **样本量不足**：首轮 10 seed 显示"ε≤40% 全 10/10、50% 崩 3/10"；
     换一组全新 seed（2000–2009）复跑，**ε=33% 就有 1/10 撞车，50% 升到 6/10**。
     → 10 seed 结论不可靠。**改用 20 seed（两组各 10 合并）**，重采全部数据。

- **最终结果（20 seed 协议，固定 1000–1009 + 2000–2009）**:

  ```
  eps% | RL: collision-free  mean|e| | P+FF: collision-free  mean|e|
    0  |     20/20           1.23 cm |      20/20            2.09 cm
   10  |     20/20           1.22 cm |      20/20            2.09 cm
   20  |     20/20           1.28 cm |      20/20            2.10 cm
   30  |     20/20           1.33 cm |      20/20            2.13 cm   <- DR 边界内
   33  |     19/20           1.31 cm |      20/20            2.14 cm   <- DR 边界 (a_max≈1.0)
   40  |     18/20           1.39 cm |      20/20            2.19 cm
   50  |     11/20           1.51 cm |      20/20            2.30 cm   <- 深度外推
  ```

- **发现（修正了 plan 的假设 H1–H3）**:
  - **H2 被证伪**：RL 并非"斜率更小"。**RL 有失效边界，P+FF 没有**——
    RL 在 ε=33%（DR 边界处）开始零星失败，50% 时降到 11/20；P+FF 全程 20/20 零碰撞。
  - **但 RL 的绝对精度全程更优**：即使 50% 外推档，RL mean|e|=1.51 cm 仍
    **显著优于** P+FF 的 2.30 cm。
  - 失效模式可解释：撞车发生在**大初始间距的追赶场景**（gap0≈0.7–0.8），
    a_max 降至 0.75（低于 DR 下界 1.0）时策略"要求的制动/加速物理上做不到"。
  - **分布内外分段得到实证**：ε≤30%（分布内）零失败且误差平稳；
    ε≥33%（越界）失败率单调上升 1→2→9（/20）。
  - 这正是 RQ4 的核心素材：**域随机化给出的是"分布内"鲁棒性；
    分布外退化不可由 DR 预测，存在明确的失效边界。**

- **验收结论**: CSV + 双曲线图产出，主指标为无碰撞率（R2）✅；
  分布边界线标于 ε=33.3%（R4①）✅；mean|e| 复用 `evaluate()` settled 口径（R4②）✅
- **与论文基准差异**: 论文 5.4 节只有"DR 开/关"两点（1.00→1.06 cm）。
  本扫描把该段展开为连续曲线，并新增"分布外失效边界"结论——
  **是论文 v2 §5.4 的实质扩充，非复现差异**。
- **产物**: `results/20260926_robustness_sweep/{robustness_sweep.csv, fig_robustness_curve.png}`；
  `robustness_sweep.py`（入库）
- **遗留问题**: 20 seed 在 50% 档仍有 ±2 的置信宽度；若发表需 50 seeds/档
  （成本极低）。ε=45/47.5 等中间档未采，失效边界可进一步细化。
- **下一步**: A3 SAC 消融（可选）或直接进入 W2 的 A4 离线 RL（d3rlpy）

---

## [2026-09-20] Phase A W1 补充 — 50-seed 复跑（评审意见 2）

- **commit**: 0bb4855
- **动机**: W1 收尾评审意见 2——20 seed 在 ε=50% 档置信宽度 ±2，
  发表需补到 50 seed（成本极低，10 分钟机时换"无需答辩"的图）
- **命令**:

  ```
  uv run python robustness_sweep.py --eps-list 0 10 20 30 33 40 50 \
      --seeds 1000..1009 2000..2009 3000..3009 4000..4009 5000..5009
  ```

- **结果（50 seed）**:

  ```
  eps% | RL: collision-free  mean|e| | P+FF: collision-free  mean|e|
    0  |     50/50           1.21 cm |      50/50            2.10 cm
   10  |     50/50           1.21 cm |      50/50            2.10 cm
   20  |     50/50           1.25 cm |      50/50            2.11 cm
   30  |     47/50 (3撞)     1.30 cm |      50/50            2.14 cm
   33  |     41/50 (9撞)     1.36 cm |      50/50            2.16 cm
   40  |     37/50 (13撞)    1.43 cm |      50/50            2.21 cm
   50  |     27/50 (23撞)    1.50 cm |      50/50            2.33 cm
  ```

- **对 20-seed 结论的修正（重要）**: 20 seed 曾显示"ε≤30% 全过、33% 起失效"；
  50 seed 显示**失效其实从 ε=30% 即开始（3/50）**——而 ε=30%（τ=0.156、a_max=1.05）
  **仍在 DR 分布内**（贴边）。**准确表述**：分布边缘已现稀有失效，越界后单调加速恶化
  （3→9→13→23/50），非"边界处突然失效"。
- **统计意义**: 50 seed 下 50% 档 27/50 = 54% 无碰撞（95% CI 约 ±14%），
  论文可安全引用；P+FF 全程 50/50 零碰撞（0/50 上限 CI ≈ 6%）。
- **验收结论**: 曲线置信度达标 ✅；RL 精度全程优于 P+FF（1.50 vs 2.33 cm @50%）
- **与论文基准差异**: 同 W1（连续曲线为论文 v2 §5.4 扩充）。
- **产物**: `results/20260926_robustness_sweep/`（CSV `n_seeds=50` + 更新图）
- **下一步**: W2（RQ2 离线 RL），plan 已起草待评审

---

## [2026-09-20] Phase A W2 — RQ2 离线 RL 对比（A4/A5）

- **commit**: 1b059ab（假设登记后）；本批为结果提交
- **plan**: `plan_W2_offline_rl.md`（review1 修订版）；review: `reviews/20260920_w2_plan_review1.md`
- **假设登记**: `research/ASSUMPTIONS.md`（H1–H3，训练前写入，阈值定死）

### 两个试点门（均通过，先于全量）

```
门 2（规则基线锚点，d3rlpy venv / gymnasium 1.0.0）:
  rule-based  overtake=10/10  collision=0 offtrack=0  t_overtake=2.0s  mean_v=0.55
  → 与主 venv 完全一致，两 gymnasium 版本对本 env 行为等价 ✅

门 1（100+ transition API 微型拟合）:
  MDPDataset 构造 OK；IQL / TD3PlusBC fit + predict 均通过
```

**环境隔离验证**：d3rlpy venv gymnasium=1.0.0、主 venv gymnasium=1.3.0（未变）、
5 个冻结文件 sha256 **全部 OK**。离线 venv 已 gitignore。

**门价值**：试点门抓到 3 个会在全量阶段浪费机时的问题——① `obs[:100]` 切片错误
（落在集中间，无终止标记，`MDPDataset` 拒绝）；② `MDPDataset` 无 `__len__`
（正解 `.episodes`/`.transition_count`）；③ d3rlpy venv 缺 matplotlib（导入链）。

### 训练

```
命令: .venv-d3rlpy/bin/python rq2_offline.py train --algo {iql,td3bc} --demos 300 --n-steps 100000
数据: demos_v1.npz（300 集 / 41936 转移，映射 A：terminals=真终态含 success，
      timeouts=timeout/failed；从 episode_reasons 重构）
设备: CPU；d3rlpy 2.8.1；均 10 epoch × 10000 steps
```

- IQL: critic_loss 收敛平稳（末值 ≈14.1）
- TD3+BC: **critic_loss 末期爆炸式增长**（13747 → 321130 → 2208362）——见遗留问题

### 评估结果（10 固定 seed 2000–2009，五类终止全审计）

**四方对照表**（名义参数 / 域随机化）：

| 方法 | 训练 | 成功率(名义) | 成功率(DR) | 碰撞 | 冲出 | t_ot | mean_v |
|---|---|---|---|---|---|---|---|
| BC（已有） | 监督 | 2/10 | — | 8 | 0 | — | 0.50 |
| **IQL** | 离线 | **10/10** | **10/10** | **0** | 0 | 2.0 s | 0.55 |
| **TD3+BC** | 离线 | **10/10** | **10/10** | **0** | 0 | **1.7 s** | 0.61 |
| PPO 微调（已有） | 在线 | 10/10 | 10/10 | 0 | 0 | 1.4 s | 0.71 |

原始输出：
```
iql    d=300 dr=False: success=10/10 collision=0 offtrack=0 lost=0 failed=0 t_ot=2.0s mean_v=0.55
iql    d=300 dr=True : success=10/10 collision=0 offtrack=0 lost=0 failed=0 t_ot=2.0s mean_v=0.54
td3bc  d=300 dr=False: success=10/10 collision=0 offtrack=0 lost=0 failed=0 t_ot=1.7s mean_v=0.61
td3bc  d=300 dr=True : success=10/10 collision=0 offtrack=0 lost=0 failed=0 t_ot=1.7s mean_v=0.61
```

逐 seed 确认两者**真超车**（t_ot 有分布 1.2–2.7 s，非恒定；IQL 1.3–2.7、TD3+BC 1.2–2.2）。

### 假设检验

| # | 假设 | 阈值 | 实测 | 判定 |
|---|---|---|---|---|
| H1 | IQL ∈ [3/10, 9/10]（介于 BC 与 PPO） | — | **10/10** | **证伪**（IQL 追平在线微调，非"介于"） |
| H2 | IQL collision ≤ 6/10 | — | **0** | **成立**（远优于阈值） |
| H3 | IQL 与 TD3+BC 成功率差 ≤ 2/10 且碰撞差 ≤ 2 | — | 差 0，碰撞差 0 | **成立**（但 TD3+BC 更快 1.7 vs 2.0 s） |

### 发现（RQ2 的核心答案）

- **H1 被证伪是本轮最重要的结果**：IQL 从**纯离线** 41936 条演示中达到 **10/10**，
  追平在线微调的 PPO。这说明**演示数据本身已足以支撑最优策略**，
  在线微调并未"超越数据"——只是 RF 收敛更快（1.4 s vs 2.0 s）。
- **安全边界**：离线 RL（IQL/TD3+BC）零碰撞，远优于 BC（8 次）——
  印证 H2：离线 RL 的保守性（Q 下界 / BC 正则）确实抑制了撞车。
- **三种方法的数据效率排序（t_ot）**：PPO 微调 1.4 s > TD3+BC 1.7 s > IQL 2.0 s
  ≈ 规则 2.0 s。IQL 基本复现了演示者（规则机），TD3+BC 略有改进。
- **DR 全组零退化**：四方法在 DR 下成功率/耗时与名义一致。

### 验收结论

- 四方对照表产出（论文 4.8 节素材）✅
- 离线 RL **训练未接触环境**（代码结构强制：train 只读 npz）✅
- 五类终止全审计 ✅；假设逐条检验 ✅（H1 证伪如实记录）

### 与论文基准差异

- 论文 4.5 节仅断言"BC 2/10 不够，需 RL 接管"。本轮**新增了 BC 之外的三方对照**，
  且发现**离线 RL 单独即可达 10/10**——这是论文 4.8 节的全新内容，非复现差异。

### 遗留问题（重要）

1. **TD3+BC critic_loss 发散**（末值 2.2e6）：虽然评估 10/10，但训练不稳定是真实
   信号，结果稳健性存疑。**需在论文中如实标注**，或补"降 lr / 加早停"复测。
2. **模型交付**：`ckpt_offline/{iql,td3bc}_d300.pt`（3.9 MB / 3.3 MB），
   sha256 `9eff21e2…b0566` / `ca720969…36cc`。已 gitignore，用目录归档 + 本记录存证。
3. 数据量扫描（R4，50/100/200/300 × 3 方法）**未跑**——本轮先出 300 档四点对照，
   扫描作为 W2 余项或 W3 前置。

- **产物**: `results/20260920_rq2_offline/`；`rq2_offline.py`（入库）；
  `requirements-offline.txt`（入库）；模型存 `ckpt_offline/`（归档）
- **下一步**: 数据量扫描（R4）→ 论文 v2 §4.8 + §5.4 修订（A6）

---

## [2026-09-20] Phase A A6 — 论文 v2 修订（§4.8 新增 + §5.4 修正 + 摘要）

- **commit**: a18f72f
- **执行内容**:

  1. **§4.8 新增**（插入 §4.7 后，`project_report_full.md:279`）：
     "模仿、离线 RL 与在线微调：一个能力-效率谱系"，含
     - 4.8.1 实验设计（30 seed 协议 + 五行对照表）
     - 4.8.2 数据效率（嵌套子集曲线 + H4 检验）
     - 4.8.3 实现敏感性（两个 BC 对照 + H6/H7）
     - 4.8.4 讨论（能力-效率谱系 + 对 4.5 节 LfD 叙事的精化）
  2. **§5.4 扩展**：新增 5.4.1"鲁棒性的连续刻画"，把"开/关两点"展开为
     50-seed 连续曲线（表 5.5），含失效边界与分布内外的修正结论。
  3. **摘要修正**："域随机化下性能完全不退化" → "**训练分布范围内**的域随机化
     下性能不退化"，并补一句 RQ2 能力-效率谱系的发现。
  4. **图 14** 归档为 `project_paper/figures_v2/图14_RQ2数据效率_v1.png`，
     入图索引表。

- **30-seed 终版五方法表**（评审 P1 要求，已兑现）:

  ```
  方法                  名义      碰撞   t_ot    mean_v
  manual-BC           5/30      25     2.3s    0.50
  d3rlpy-BC          30/30       0     2.1s    0.53
  IQL                30/30       0     2.1s    0.53
  TD3+BC (lrhlf)     30/30       0     2.0s    0.26
  PPO 微调           30/30       0     1.4s    0.70
  ```
  DR 组四方法均 30/30（manual-BC 6/30），无退化。

- **验收结论**: §4.8 零占位符（全文 `grep 待填` = 0），数字全部可溯源 ✅
- **与论文基准差异**: §4.8 为全新章节（论文 v1 无）；§5.4 的"完全不退化"
  经 W1 数据修正为"分布范围内不退化 + 分布外有失效边界"。
- **遗留问题**:
  - TD3+BC 非单调（N=50/300 崩，100/200 好）已如实写入 4.8.2；
  - manual-BC 机制（H7）未定位，记为 future work；
  - 报告 §6（调试经验）等章未同步 A 阶段新发现（可选）。
- **产物**: `project_paper/project_report_full.md`（1128 行）、
  `results/20260920_rq2_offline/`（6 文件）、`plot_rq2_efficiency.py`
- **下一步**: 报告评审 → 开源发布（Phase B）或收尾

---

## [2026-09-20] Phase B W3 — 仓库公开化与发布

- **commit**: da6da2c（README/CITATION）、84342bd（Makefile）
- **目标**: 仓库公开，建立社区基线（roadmap Phase B W3，零新实验，纯包装）

### 执行内容

| 任务 | 结果 |
|---|---|
| B1 README 重写 | ✅ AGV 双重叙事、11/11 复刻徽章、结果表、快速开始、仓库结构、评估协议说明 |
| B2 叙事包装 | ✅ 双叙事分层：AGV（应用域）与 TI 杯（出处）**均为真**，README 显式说明二者指同一系统 |
| B3 一键运行 | ✅ `Makefile`（`make check` / `demo` / `reproduce`）；`make demo` **实测跑通**（~5 min 全流程，短步数 smoke test） |
| B4 发布 checklist | ✅ LICENSE(MIT) 已有；✅ 新增 CITATION.cff；✅ `.gitignore` 含 `!demos_v1.npz` 例外；✅ Release v1.0.0 + 3 资产 |

### Release v1.0.0

- **URL**: https://github.com/2572873335/car-rl/releases/tag/v1.0.0
- **仓库可见性**: **public** ✅
- **资产**（GitHub 端 sha256 与本地逐一吻合，证明上传无损）:

  | 资产 | 大小 | sha256 |
  |---|---|---|
  | follow_stage2_final_v1.zip | 452390 | `4231a613…9caa2` ✅ |
  | overtake_final_v1.zip | 460319 | `ebf0b792…62f12` ✅ |
  | demos_v1.npz | 1383277 | `4b31c47e…f429` ✅ |

- **tag**: `v1.0.0`（指向 main）；另有里程碑 tag `milestone/phaseA-results`

### 推送记录

```
20bfafc..27ee67d  main -> main   (Phase A: 17 commits)
27ee67d..84342bd  main -> main   (Phase B W3)
milestone/phaseA-results (new tag)
v1.0.0 (release tag, created via API)
```

### 网络处置（环境问题，非项目内容）

WSL 的 `github.com`/`api.github.com` 被 Windows 侧 GitHub 加速工具经
hosts 污染至 `127.0.0.1`。处置：git 走 SSH over 443（`ssh.github.com`，
不依赖 hosts）；Release 经加速器网关 `172.23.192.1` + `curl -k` 调 API 完成
（gh CLI 因加速器自签证书无法用）。**临时 hosts 映射已从备份还原**，
WSL 恢复原状。

### 验收结论

Phase B W3 完成：仓库公开、README/CITATION/Makefile 就位、Release v1.0.0
含 3 资产且哈希校验通过 ✅

### 遗留问题 / 安全提醒

- **⚠️ 安全**: 本次使用的 GitHub PAT 已在会话中明文出现，**应立即在
  GitHub 设置中撤销**（Settings → Developer settings → Personal access tokens）。
- **B5/B6（内容营销）**: 博客与社区发布需人工账号操作，未执行。
- **B7 数据资产**: `demos_v1.npz` 随 Release 发布即完成 ✅。

- **下一步**: W4 内容营销（博客①②草稿）或收尾

---

## Phase C W5 — 自博弈探针（RQ3 硬门）

> 性质：**机制探针**（roadmap §Phase C W5 的 3 天硬门）。本轮**未产出研究结论**，
> 产出的是"这条路能否走通"的判定与判据设计。所有实验在 `/tmp` 下以一次性脚本
> 执行，**仓库冻结文件零改动**（5 文件 sha256 全部 OK）。

### [2026-09-20] RC.1 `[eval]` 机制接通性 pilot gate

- **命令**: `uv run python /tmp/_pilot_gate.py`（PYTHONPATH 指向仓库）
- **目的**: 验证 SB3 自定义 `VecEnv` + 共享参数能否驱动两车；属评审建议的
  "最廉价证伪"前置。
- **原始输出（关键行）**:

  ```
  n_worlds=4 -> num_envs(rows)=8
  per-agent obs space = Box(-inf, inf, (6,), float32)
  policy features_dim = 6  (MUST be 6)
  shared policy params = 35205
  features_dim assert PASSED -> obs-flattening pitfall avoided
  ```

- **验收结论**: 机制**接通** ✅（共享参数确认：同 obs → 同动作）
- **踩坑记录**（两条真实机械坑，已写入 plan §1.5/§1.6）:
  1. `observation_space=Box(shape=(2,4))` → `RuntimeError: mat1 and mat2 shapes
     cannot be multiplied (2x4 and 8x64)`——SB3 会**摊平前导维**；
     正解是声明**单 agent 形状**，用 `num_envs=2N` 表达多行。
  2. `SubprocVecEnv` 顶层直接实例化 → `EOFError: unexpected EOF`
     （forkserver 反复 re-import）；正解是 `if __name__ == "__main__":` 守卫。
- **产物**: `results/20260920_phaseC_probe/scripts/_pilot_gate.py`

### [2026-09-20] RC.2 `[eval]` 判据体检——发现两处判据级错误

- **命令**: `/tmp/_test_return_signal.py`、`/tmp/_rule_vs_rule.py`、`/tmp/_chase_redflag.py`
- **原始输出（关键行）**:

  ```
  policy           mean(r0+r1)  collision%  offtrack%   predicted
  zero                     0.0        0.0%       0.0%        -0.0
  random                -825.0       82.5%       0.0%      -825.0
  rule                     0.0        0.0%       0.0%        -0.0

  CHECK 1: delta0 at reset over 20 episodes: mean=+0.428 m, min=+0.180
           all positive (car 0 always starts ahead)? True

  RULE vs RULE: episodes with ZERO lead change: 40/40
                did either car USE the inner lane? 0/40 episodes
  ```

- **验收结论**: 两处判据级错误被**在动手前**拦下：
  1. plan 原称"对称零和 ⇒ 平均回报恒 0，不可作指标"——**证伪**：
     `mean(r0+r1) = −(1000·p_coll + 200·p_off)` 逐行吻合，回报**携带失败率信号**；
  2. plan 原 P2"对规则对手胜率 ≥50%"——**饱和**：车 0 固定领先 0.43 m，
     未训练策略即 20/20；且**规则机 40/40 局零次领先易手、零次用内圈**，
     **根本不是博弈对手**，判据不可修复。
- **产物**: 同名脚本 + `findings_phaseC_selftest.md`

### [2026-09-20] RC.3 `[train]` 完整联合自博弈（探针主实验 A）

- **命令**: `/tmp/_run_real_probe2.py`，8 worlds = 16 rows，`n_steps=256`
- **超参**: 沿用冻结 PPO（`MlpPolicy[128,128]`、lr 3e-4、batch 512、γ0.99、
  GAE 0.95、clip 0.2、ent_coef 0.01、seed 0）；`device='cuda'`（后经复测改为 cpu，见 RC.5）
- **更新次数**: 1,280,000/(16×256) = **312.5** ✅（满足铁律 4，虽探针豁免）
- **原始输出（趋势摘录）**:

  ```
        steps   mean_ret  ~collision%  entropy_loss   value_loss
        1,728     -740.0        74.0%
        2,928      -10.0         1.0%     ← 早期改善
        4,128        0.0         0.0%
      ...
    1,247,552     -880.0        88.0%     -3.369        1.97e+04
    1,254,000    -1000.0       100.0%     -3.370        1.88e+04
    1,259,824     -880.0        88.0%     -3.371        8.14e+03
  ```

- **验收结论**: **不收敛** ❌——碰撞率 4k 步降至 0% 后**反弹并钉死 88–100%**；
  回报钉在 −880~−1000；熵**不降反升**（2.84 → 3.37 nats，即策略越来越随机、
  未收敛到确定行为）。训练 814 s。
- **→ P1 失败，P2（三对照全部饱和）失败**
- **产物**: `results/20260920_phaseC_probe/probe_run2_joint_selfplay_trend_raw.txt`

### [2026-09-20] RC.4 `[train]` 冻结一方（探针主实验 B，roadmap"联赛退化档"）

- **命令**: `/tmp/_step1_frozen.py`——对手 = 20 次更新后的快照，只训另一车
- **更新次数**: 200（409,600 步，8 行）
- **原始输出**:

  ```
    episodes  mean_return  collision%
          51       -500.1      100.0%
       30432          0.3        0.0%
       98512         -0.3        0.0%
      166592        -60.4       12.0%
      404872       -120.8       24.0%
  ```

- **验收结论**: **学习发生了** ✅——碰撞率 **100% → 24%**（多数时段为 0%），
  回报 −500 → 近 0。训练 832 s。
- **→ P2 成立**（冻结对局下碰撞率相对下降 ≥50%）
- **产物**: `results/20260920_phaseC_probe/probe_run3_frozen_opponent_raw.txt`

### [2026-09-20] RC.5 `[eval]` 基准测量——device 与并行扩展

- **命令**: `/tmp/_bench_parallel2.py`、`/tmp/_device_matrix.py`
- **原始输出**:

  ```
  [并行扩展，规则策略驱动冻结 OvertakeEnv]
    n_envs= 1:  1.00x (efficiency 100.0%)
    n_envs= 8:  2.89x (efficiency  36.1%)
    n_envs=16:  3.40x (efficiency  21.3%)

  [device × VecEnv 设计，机器空闲]
    [A] SubprocVecEnv (8 进程):   cpu 2,587  vs cuda 1,845 steps/s -> cpu 快 40%
    [B] 单进程 self-play VecEnv:  cpu 2,386  vs cuda 1,881 steps/s -> cpu 快 27%
  ```

- **验收结论**: 并行扩展**远非线性**（8 路仅 36% 效率）；**CPU 在两种设计下均更快**，
  故 `device='cpu'` 正确。
- **⚠️ 重要教训（本轮新增纪律）**: device 结论**曾被并发负载污染而反转**——
  首测（同时有 1.28M 步训练在跑）得"cuda 快 1.9×"，机器空闲后复测得如上表。
  **基准测试必须在无其他训练进程时运行，并在记录中注明"机器空闲"状态。**
- **产物**: 同名脚本 + `findings_phaseC_selftest.md` §F6/§F8

### 阶段结论

Phase C W5 探针**已回答**（当天，非 3 天）：**障碍是非平稳性，不是机制**。
机制接通良好；完整自博弈不收敛，而**冻结一方即学会**。
→ 探针主路径定为**冻结一方的联赛模式**；完整自博弈收敛问题降级为 W6–W9 研究内容。
详见 `plan_phaseC_probe_v2.md` §2.4 与 `research/ASSUMPTIONS.md` P1–P3。

**评审流程价值**：本轮为评审流程**第五次在提交前拦截真问题**。
若 B1（回报恒 0）+ B2（熵判据方向反）带入执行，会在 Day 3 得到
"回报不动 + 熵没降"的**双假阴性**，**错误判定 RQ3 失败并提前启动 Phase D**。

### 遗留问题

- **`DATA_MANAGEMENT.md` §10 缺失**：全文仅 §1–§9，但被 12+ 文件引用为
  "评审流程"依据（`AGENT_HANDOFF`/两份 plan/`RUNLOG`/`ASSUMPTIONS`/三份 `reviews`）。
  流程本身真实存在，但章节未写。**建议补写或统一改引用**。
- **报告内圈长度口径**：报告 §5.6/§11.5 写"4.54 m vs 5.49 m"，实测
  `outer=5.4849 / inner=4.5424`，四舍五入应为 **5.48**。
- **冻结文件哈希表待补**：`selfplay_env.py`/`train_selfplay.py` 尚未创建；
  创建后须补入 `DATA_MANAGEMENT.md` §8。

---
## Phase B W4 — 内容营销（KPI 基线）

### [2026-09-22] RB.1 `[eval]` 博客发布与 KPI 基线回填

- **动作**: 博客①《「跟车」问题的 RL 解法：从零到 1.0 cm》发布至知乎
  （`https://zhuanlan.zhihu.com/p/2085539647359284278`，由项目负责人发布）
- **原始输出（可测部分，本机实测非估算）**:

  ```
  # 网关路线（github 域名被加速器污染，走 ssh/API 网关 172.23.192.1）
  $ curl -sk --resolve api.github.com:443:172.23.192.1 \
      https://api.github.com/repos/2572873335/car-rl
    stargazers_count : 1
    forks_count      : 0
    watchers_count   : 1
    subscribers_count: 0
    open_issues_count: 0
    created_at       : 2026-09-19T11:40:15Z
    pushed_at        : 2026-09-21T16:46:07Z

  $ curl -sk .../releases/latest
    tag : v1.0.0   published : 2026-09-20T03:44:14Z
    assets: demos_v1.npz(1383277) follow_stage2_final_v1.zip(452390)
            overtake_final_v1.zip(460319)
  ```

- **KPI 基线（**12 月考核起算点**）**:

  | 指标 | 值 | 采集方式 | 时间 |
  |---|---|---|---|
  | GitHub star | **1** | GitHub API（网关路线） | 2026-09-22 |
  | GitHub fork | **0** | 同上 | 2026-09-22 |
  | watchers / subscribers | 1 / 0 | 同上 | 2026-09-22 |
  | Release v1.0.0 资产 | 3 个（哈希已在 §9 存证） | 同上 | 2026-09-22 |
  | **知乎阅读数** | **【待负责人回填】** | 知乎后台（agent 无账号，不可测） | — |
  | 知乎赞同/收藏/评论 | **【待负责人回填】** | 同上 | — |

- **验收结论**: 博客①已发布 ✅；可测 KPI 已回填 ✅；
  **知乎侧三项数值须由负责人补入上表**（agent 无该账号权限，不代填、不估算）
- **与 G2 决策门的关系**（`research/roadmap.md` W4）:
  > "star < 10 不焦虑（内容传播有滞后），但需在 RUNLOG 记录基线数据"
  **→ 本条目即为该基线**。发布仅 2 天，**1 star 属预期，不作好坏判断**。
- **产物**: 本条目；博客草稿 `blog/blog_01_following_rl_1cm.md`（已发布版本）
- **遗留 / 下一步**:
  - 博客②（工程纪律）**尚未发布**，与①同源同规格，可直接复用本条目格式；
  - 发布渠道 B6 计划中的 V2EX / Reddit r/reinforcementlearning **尚未投递**
    （英文摘要已在草稿内备好）；
  - **D0 场景保真度审计**已立项（见下一条），硬件采购顺延至 D0 通过。

---


## [2026-09-22] D0 文档链修复 + 口径实测（无训练，无冻结文件改动）

- **命令**: 见下「产物」逐项
- **目的**: 修复 `c0b7b7d` 提交信息声称但**从未落地**的 plan 修改；
  并对 D0 数字口径做首次实测标定
- **机器状态披露（F8）**: 跑前 `pgrep -af 'uv run python'` 仅本批脚本自身；
  全部为**确定性计数**（`mean|e|`、碰撞数），同 seed 可复现

### 1. 发现：提交信息与落地事实不符

`c0b7b7d`（09:41）声称「plan v2 加入 wrapper 硬约束」+「新增 §8」，
但 `git show --stat` 显示该提交**只改了 `README.md` 与 `_d0_step0_wrapper_probe.py`**。

```
$ git show c0b7b7d:plan_D0_scenario_audit_v2.md | grep -c wrapper
0
$ git log --all -S 'D0.5'      # 空
```

**根因**：编辑脚本 `_d0_wrapper_mandate.py` 只存在于 `/tmp`（09:39），
目标文件在仓库；两副本未同步 ⇒ 编辑静默丢失，而提交信息按**意图**书写。
**这正是已登记的两个坑的叠加**（「写完的文档提交后被旧副本覆盖」+
「改一处另一份没改」）。

### 2. 实测：D0 数字的聚合口径未声明，同一格差 41%

```
$ uv run python results/20260922_D0_scenario_audit_scripts/_d0_reconcile_conventions.py
  (30 seed, true-cm = 100*|e|; whole = 全回合, settled = e[20%:])

  grid                    RLwhole  RLsett  PFFwhole  PFFsett  winner(sett)
  d=0.2 v=0.3 consta        2.48    0.31     4.59     1.05           RL
  d=0.2 v=0.5 consta        3.61    1.14     5.18     1.27           RL
  d=0.5 v=0.55 consta       2.52    1.40     4.20     2.41           RL
  d=0.5 v=1.0 consta        5.63    3.99     5.09     2.87         P+FF
  d=0.5 v=1.0 sinuso        4.93    3.44     5.02     2.87         P+FF
  d=0.5 v=1.0 brake         4.29    3.52     4.30     2.75         P+FF
  d=1.0 v=1.0 consta        4.74    3.82     5.21     3.10         P+FF

  follow_env.py sha256[:16] before/after: 941c447b969c2fa1  untouched=True
```

- **混用口径的是作者侧**：README / plan v2 §1.2 / `c0b7b7d` 提交信息用 **whole** 口径
  （5.63/5.09）且未声明；`review1` 用 **settled** 口径（3.99 ≡ 7.98 obs-cm）**自洽**
  （grep 证实 review1 全文 0 次出现 5.63/5.09）。
  ⇒ 同一格两个数字，根源是作者侧未声明，**不是评审报告内部矛盾**（初稿此说有误，已撤回）。
- 仓库权威口径 = `train_ppo.py:90`（`obs[0]`×100 ≡ 200|e|，**且跳前 20%**）
  = **obs-cm + settled**。D0 既有脚本（`_verify_r4_grid.py`、
  `_d0_step0_wrapper_probe.py`）用 **true-cm + whole**，与仓库数字**不可并列**。
- **反转范围比 README 现文写的更广**：settled 口径下 v=1.0 **全部格子** P+FF 胜
  （含评审 R4 表标「RL 微弱胜」的 sinusoid 与 brake 格）。

### 3. 落点修复（§5b / F19）

`/tmp` 抢救入库（`archive/d0_owner_ruling_text/`，附 README）：

| 文件 | 内容 |
|---|---|
| `_d0_wrapper_mandate.py` | **owner 裁决原始文本**（wrapper 硬约束 + §8 + D0.5–D0.7 真阈值） |
| `_probe_d0_stageA.py` | 阶段 A 探针（已被 wrapper probe 取代） |
| `_commit_d0v2.txt` | `7a30995` 提交信息草稿 |
| `_d0_wrapper_probe_tmpcopy.py` | /tmp 双副本的证据 |

**未找到**：`_d0_review3`（评审 R3/R4 引用）仓库与 `/tmp` 均无落点，疑随重启蒸发；
其作用已由 `_d0_reconcile_conventions.py` 替代（口径声明更明确）。

### 4. 文档更新

- 新增 `plan_D0_scenario_audit_v3.md`：**正式吸收 owner wrapper 裁决**（v2 §1.1/§4/§7
  的「改 `D_DES`」作废）+ 补 §8 regime-extension（**owner 真阈值**，非交接者起草）
  + 新增 §2.0 聚合口径声明；
- `research/ASSUMPTIONS.md` 追加 **D0 预登记块**（D0.1–D0.7 + 口径声明 + 边界警报程序）；
- `archive/d0_owner_ruling_text/README.md`。

- **验收结论**: 文档链已一致 ✅；口径已标定并声明 ✅；
  **plan v3 待 review2**（§10 流程：plan → review → execute）⇒ **扫描未开跑**
- **产物**: `plan_D0_scenario_audit_v3.md`、
  `results/20260922_D0_scenario_audit_scripts/{_d0_reconcile_conventions.py,_r4_regrid_30seed.txt}`、
  `archive/d0_owner_ruling_text/`、本条目
- **遗留 / 下一步**:
  - **plan v3 送 review2**（`reviews/20260922_D0_review2.md`），通过后才开跑 9 格扫描；
  - **D0.2 贴线（7.98 vs 8.00）须补 30→100 seed**，不放宽阈值；
  - ~~**README 口径错配**~~ **已修**：operating-range 段已改为
    obs-cm + settled 口径（`200·|e|`，跳前 20%），并把反转范围从「单格」改为
    「every cell tested」（与实测一致）；链接已指向 v3；
  - 0.55 分布内边界的**来源须由 review2 裁定**（v2 遗留确认点 3）。

---

## [2026-09-22] D0 review2 修订落地 + B1 边界独立复现（无训练，无冻结文件改动）

- **命令**: `uv run python results/20260922_D0_scenario_audit_scripts/_d0_review2_boundary_100seed.py`
  等；逐项见下
- **目的**: 处理 review2（`reviews/20260922_D0_review2.md`，裁决 **Approve with amendments**）
  的 4 项拦截级 B1–B4 + 6 项重要问题 I1–I6
- **机器状态披露（F8）**: 跑前 `pgrep -af 'uv run python'` 仅本批脚本自身；
  全部确定性计数，同 seed 可复现

### 1. B1（★ 最重要）独立复现：D0.2 贴线是**结构性**，非抽样噪声

复现 `_d0_review2_boundary_100seed.py`，与评审**逐位吻合**：

```
d=0.50 v=1.00 constant, settled obs-cm：
  seeds 2000-2099: agg_mean=7.973 std=0.160 max=8.386  ep>8.00: 39/100  coll=0/100
  seeds 3000-3099: agg_mean=7.946 std=0.185 max=8.337  ep>8.00: 36/100  coll=0/100
  seeds 4000-4099: agg_mean=7.954 std=0.177 max=8.544  ep>8.00: 37/100  coll=0/100
  seeds 5000-5099: agg_mean=7.979 std=0.188 max=8.463  ep>8.00: 44/100  coll=0/100
  P+FF 100 seeds:  agg_mean=5.731 std=0.120 coll=0/100
```

- **30 seed 的 7.98 与 100 seed 的 7.97 一致** ⇒ 结构性贴线，**补 seed 不能判定**；
- 按「100 seed 聚合均值」判 **D0.2 通过（7.95–7.98）**；按「单回合」判 **未通过（36–44% 越线）**
  ⇒ **plan 原文未定义聚合单位 = 不可判**（review1 R3 同型复发）。
- **修正**：plan v3 §2.2 新增**聚合单位定义**（= 100 seed 的 per-seed settled mean 的均值，
  判据作用于该值 + 报 ±SE；单回合越线不作判负依据）。**阈值 8 不放宽**，余量 ~0.03 cm 明标。

### 2. B2 「更正只做了一半」——本次自查确认并修复

`ade4041` 撤回「评审混用口径」时**只改了 blockquote**，漏改紧邻的事实 3 括号，
致同节「review1 自洽」与「review1 混用口径」并存。**grep 实证**：`微弱` 在 review1 出现 **0 次**。
⇒ 已删该误归因，改记「作者侧 plan v2 §1.2 的 whole 口径标注」。
**教训（已入记忆）**：撤回一个说法须 grep **全部变体**，改完做矛盾自检。

### 3. B3 wrapper `D_DES` 异常路径泄漏——**已复现并加固**

```
(a) 正常交叠 RL→close→P+FF：干净（fe.D_DES 归 0.20）
(b) 异常抛出且 close() 未执行 → fe.D_DES 滞留 0.50
    raw env 追错设定点：terminal gap=0.463（应为 0.20）   ← 泄漏确凿
(c) baseline_action 不读 D_DES（附带的伪风险，已证伪）
```

⇒ plan v3 §1.1.1 新增**强制 try/finally + 每 run 后断言 `fe.D_DES == 0.20`**；
**并已加固本次自己的落盘脚本**（`_d0_reconcile_conventions.py`、`_d0_step0_wrapper_probe.py`）。
（评审的探针脚本保留原样——它们是该发现的证据本身。）

### 4. B4 §8.2 陈旧 caveat——已删

「D0.5–D0.7 为交接者起草」系入库存档前的旧文字；`_d0_wrapper_mandate.py` §8.3
**逐字含有** D0.5–D0.7 ⇒ 即 **owner 原文**。已改为「出处已核对（review2 E）」。

### 5. I1–I6 一并落地

- **I1**：0.55 锚定改用**实证**（训练 sinusoid 瞬时领车速度达 **0.70 m/s**，0.55 在已见带内）；
- **I2**：reconcile 脚本网格扩至 **11 行**，补回 review1 R4 引用的 `(0.20, 1.00)`；
- **I3**：§2.4「正值证据」与 D0.2 判定**解耦**（前者是事实陈述，后者独立裁定）；
- **I4**：`MAX_GAP=2.0` 不随 `d_des` 变 ⇒ lost 窗口漂移，已在 §3 显式说明；
- **I5**：§2.5 定义「9 格」= **(d,v) 二维网格**、behavior 取 `constant`，变体单列不计入；
- **I6**：§8.1 增 **Step 0.5**（`SubprocVecEnv` 子进程侧注入 `v_set` 的冒烟验证）。

### 6. 诚实条款（review2 四·3）

训练已见 **0.70 m/s** 瞬时领车速度 ⇒ **v=1.0 只高 43%，非突变**。
报告/README **不得**表述为「1.0 m/s 是截然不同的分布外速度」，已写入 plan §2.3 与 ASSUMPTIONS。

- **验收结论**: review2 的 4 项拦截级 + 6 项重要问题**全部处理** ✅；
  plan v3 判据**已可判**（聚合单位已定义）✅；脚本已加固 ✅
- **产物**: `reviews/20260922_D0_review2.md`（评审）、
  `results/20260922_D0_scenario_audit_scripts/{_d0_review2_leak_probe.py,_d0_review2_boundary_100seed.py,_d0_review2_degenerate_probe.py}`（评审落盘）、
  `_d0_review2_boundary_repro.txt`（本次复现）
- **遗留 / 下一步**:
  - **§8.1 是否触发铁律 2 的裁定**（review2：**不触发铁律 2，正确触发铁律 4**；
    新 ckpt 须如实标 v2）——已采纳；
  - **可开跑 9 格扫描**（review2 已 Approve；B1–B4 已修完）；
  - **`blind(+1)` 在 d=0.5 v=1.0 20/20 撞车**可作 §10.4 退化反例（评审附带发现）；
  - `_d0_review3` 与 `plan_D0_scenario_audit.md`(v1) 两处落点缺口已裁定并案处理。

---

## [2026-09-22] D0 场景保真度扫描（11 格 × 3 策略 × 100 seed = 3300 回合）

- **命令**: `uv run python results/20260922_D0_scenario_audit/d0_sweep.py`
- **规模**: 11 格（9 格核心 (d,v) 网格 + (0.50,1.00) 的 sinusoid/brake 变体）× {RL, P+FF, P} × 100 seed
- **耗时**: 1504 s
- **机器状态披露（F8）**: 跑前 `pgrep -af 'uv run python'` 为空；全量单进程独占
- **口径**: settled（`e[20%:]` 均值）+ obs-cm（`200×|e|`），仓库惯例；同时记录 whole
- **冻结文件**: `follow_env.py` sha256 `941c447b969c2fa1…` **前后一致**（wrapper 覆写 + try/finally + 断言）

### 主结果（settled obs-cm）

```
v<=0.50（训练带内）: RL 全胜，优势显著
  d=0.20 v=0.30  RL=0.616  P+FF=2.089  -> RL
  d=0.50 v=0.30  RL=0.530  P+FF=3.523  -> RL
  d=0.50 v=0.55  RL=2.788  P+FF=4.809  -> RL
  d=1.00 v=0.30  RL=0.604  P+FF=2.182  -> RL
  d=1.00 v=0.55  RL=2.775  P+FF=3.737  -> RL
v=0.55 边界: 混合
  d=0.20 v=0.55  RL=2.845  P+FF=2.673  -> P+FF   <- D0.4 失守格
v=1.00（分布外）: 全部 P+FF 胜
  d=0.20 v=1.00  RL=7.422  P+FF=3.300  -> P+FF
  d=0.50 v=1.00  RL=7.973  P+FF=5.731  -> P+FF
  d=1.00 v=1.00  RL=7.653  P+FF=6.205  -> P+FF
  d=0.50 v=1.00 sinusoid  RL=6.846  P+FF=5.738  -> P+FF
  d=0.50 v=1.00 brake     RL=7.237  P+FF=5.545  -> P+FF
```

### 判据判定

| # | 阈值 | 实测 | 判定 |
|---|---|---|---|
| D0.1 | RL ≤ 5 | 最大 2.788 | ✅ 通过 |
| D0.2 | RL agg ≤ 8 且零碰撞 | 7.973 / 7.653，零碰撞 | ✅ 通过（余量 0.027，1.7σ） |
| D0.3 | 先验划界 | 定义性 | ✅ |
| D0.4 | 分布内 6 格全不劣 | **5 胜 1 负** | ❌ **未通过** |

**D0.4 未通过（如实报告）**：`(d=0.20, v=0.55)` RL 2.845 > P+FF 2.673，
差 −0.172 obs-cm（**12.4σ**，统计可分辨但量级可忽略；与 v=1.0 的 1.4–4.1 **不同量级**）。
按预登记「不许跑完再划范围」，**判未通过**，划界疑虑单独记录不改判。

### 终止原因全审计（铁律 5）

RL 与 P+FF **全部 11 格 100% timeout、零碰撞/零 offtrack/零 lost**；
P（退化基线）多格 100% collision（0.20/0.30、0.50/0.30），1.0 档 71/48 次撞车。

- **验收结论**: D0.1/D0.2/D0.3 **通过**；**D0.4 未通过**（1/6 分布内格失守）。
  核心结论：**RL 精度优势限于 v ≤ 0.50；v=0.55 已现边界；v=1.0 全面反转。**
- **产物**: `results/20260922_D0_scenario_audit/{raw.json,summary.csv,config.txt,fig_D0_envelope_v1.png,d0_sweep.py}`、
  `docs/scenario_audit.md`、`research/ASSUMPTIONS.md`（D0 判定回填）
- **遗留 / 下一步**:
  - **§8 regime-extension**（D0.5–D0.7，owner 阈值）：扩训练分布到 U(0.25,1.50)、5M 步，
    验证优势是否在 AGV 区恢复；
  - **README 措辞**：现文「at 1.0 m/s」结论**成立**，但可补 v=0.55 边界的实测；
  - D0.4 失守格及其「分布内 vs 边界」划界，交 §8 一并讨论（不改判据结论）。

---

## [2026-09-22] D0 regime-extension 重训 + D0.5–D0.7 判定（**未通过，但被未收敛混淆**）

- **命令**:
  ```
  uv run python train_regime_ext.py --stage stage1 --timesteps 2500000 --n-envs 32 --seed 0 --easy --out ckpt/follow_stage1_v2_speedext.zip
  uv run python train_regime_ext.py --stage stage2 --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt/follow_stage1_v2_speedext.zip --out ckpt/follow_stage2_v2_speedext.zip
  uv run python results/20260922_D0_regime_ext/eval_d05_d07.py
  ```
- **更新次数（铁律 4）**: stage1 305、stage2 610 ✅
- **注入范围**: `v_set ~ U(0.25, 1.00)`（Step 0.5 实测修正，非 owner 初拟的 1.50）
- **机器状态披露（F8）**: 各跑前 `pgrep -af 'uv run python'` 为空
- **冻结文件**: `follow_env.py` sha256 前后一致（wrapper 注入）

### 判定：D0.5 / D0.6 / D0.7 **全部未通过**

```
D0.5 (v=1.0 格 v2 <= P+FF):     FAIL  — 5 格全负
      d=0.20 v2=8.373 P+FF=3.300
      d=0.50 v2=9.126 P+FF=5.731
      d=0.50 sinusoid v2=7.816 P+FF=5.738
      d=0.50 brake    v2=8.094 P+FF=5.545
      d=1.00 v2=8.818 P+FF=6.205
D0.6 (v<=0.55 不劣 v1):         FAIL  — 最差 1.32x (d=1.00 v=0.30)
D0.7 (9 格 >=7):                FAIL  — 5/9
```

### ★ 但结果被「未收敛」混淆 ⇒ H-D0.5 **未测**，非「证伪」

| 证据 | v1 | v2 | 读法 |
|---|---|---|---|
| 每格对比 | — | **每格都更差**（含 v=0.30，两分布内均有） | 非分布效应 |
| 策略 std（stage2 轨迹） | 0.761→1.40 稳定 | **0.814→6.13 单调发散** | 策略退化为近随机 |
| 确定性 \|a\| 均值 / \|a\|>0.9 占比 | 0.681 / 47.7% | **0.882 / 80.1%** | 动作空间饱和 |
| v=0.30 settled err (obs-cm) | 0.613 | 0.816 | 分布内格子也退化 |

**根因定位（课程缺陷）**：`regime_ext_env.make_env(easy=True)` 的 `easy`
**不改速度带**——`RegimeExtWrapper` 恒用 `U(0.25,1.00)`。故 **stage1「easy 课程」
已跑满宽速度带**，课程在速度轴上是空的（stage1 末 std 0.824 vs v1 的 0.654）。

- **验收结论**: D0.5–D0.7 按预登记**记为未通过**；但**该失败被未收敛混淆**，
  **不得**据此刻画「速度上界」或否定 H-D0.5。**须先修课程再判。**
- **诚实条款适用性说明**：plan §8.3 的 honesty clause（「修复失败 ⇒ 报为量化速度上界」）
  **本轮不适用**——该条款针对**干净的修复失败**，本轮是**训练未收敛**，二者不同。
- **产物**: `ckpt/follow_stage1_v2_speedext.zip`、`ckpt/follow_stage2_v2_speedext.zip`、
  `results/20260922_D0_regime_ext/{train_stage1.log,train_stage2.log,eval_d05_d07.py,eval_summary.csv,eval_raw.json,eval_config.txt}`
- **遗留 / 下一步**:
  - **v3：速度轴真正的课程**——stage1 用 `U(0.25,0.50)`（同 v1），stage2 放宽到 `U(0.25,1.00)`；
  - v1 审计模型 `ckpt/follow_stage2_final_v1.zip` 全程未动（sha256 `4231a613…` 已核）。

---

## [2026-09-22] regime-extension 发散诊断（v3 课程修复不足 + 高速带探针）

- **命令**:
  ```
  uv run python train_regime_ext_v3.py --stage stage1 --band narrow --timesteps 2500000 --n-envs 32 --seed 0 --out ckpt/follow_stage1_v3_speedext.zip
  uv run python train_regime_ext_v3.py --stage stage2 --band wide   --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt/follow_stage1_v3_speedext.zip --out ckpt/follow_stage2_v3_speedext.zip
  uv run python probe_highonly.py          # 高速带 U(0.80,1.00)，1M 步
  uv run python /tmp/_diag_wideband.py
  uv run python /tmp/_probe_v3_authority.py
  ```
- **更新次数（铁律 4）**: v3 stage1 305、v3 stage2 610 ✅；高速探针 123（**短探针，不产出结论**）
- **冻结文件**: `follow_env.py` sha256 全程未变（wrapper 注入）

### 1. v3（真正的速度轴课程）**仍发散** ⇒ 课程不是根因

| 跑次 | stage2 band | std 轨迹 | 结论 |
|---|---|---|---|
| v1 | U(0.25,0.50) | 0.761 → **1.40** | 稳定 |
| v2 | U(0.25,1.00) | 0.814 → **6.13** | 发散 |
| **v3** | U(0.25,1.00)（**窄 stage1**） | 0.761 → **3.98** | **仍发散** |

- v3 stage1 窄带收敛良好（reward −89.2、std 0.686，对照 v1 的 −88 / 0.654）⇒ **课程修复本身有效**；
- 但 stage2 放宽后仍发散 ⇒ **stage1 课程不是根因**（v2 的诊断为「课程空」只解释了一半）。

### 2. 诊断：不是「奖励尺度」，**高速段控制权限收缩**是结构特征

```
奖励 gap 项跨速度的均值差：v=0.30 −1.12 / v=0.55 −1.34 / v=1.00 −2.00（仅 1.8×，非数量级）
v_cmd 饱和率（P+FF）：v=0.30 0.0% → v=1.00 6.0%
控制权限（v_cmd = clip(v_l + 0.8a, 0, 1.3)）：
   v_l=0.30 → 最多提速 3.67×       v_l=1.00 → 最多提速 1.30×
```

⇒ **高速段纠错权限收缩是冻结先验的结构属性**（`v_cmd = v_l + 0.8a` + `V_MAX=1.3`）。

### 3. 高速带探针（**证据不足，仅提示**）

`probe_highonly.py`：band **U(0.80,1.00)**、**仅 123 updates**：
```
std 0.761 → 1.41（对照 v1 610 updates 后 1.40）
```
- **提示**：「纯高速窄带」未出现宽带的发散 ⇒ **「高速 regime 本身」不是充分原因**；
- **⚠ 但探针只跑 123 updates（v1/v2/v3 均为 610）**，**不足以排除**「发散需更长时间显形」。
  **不得**据此断言高速带安全。**要定论须跑满 610 updates。**

### 4. 已排除 / 未排除

| 假设 | 状态 |
|---|---|
| stage1 课程空（v2 的诊断） | **已排除为充分原因**（v3 修了仍发散） |
| 奖励平方惩罚尺度爆炸 | **证据不支持**（跨速度仅 1.8×，非数量级） |
| 高速 regime 本身 | **部分排除**（窄高速带 123 updates 未发散，但证据不足） |
| **频带宽度/异质性本身**（低+高混合） | **嫌疑最大，未验证** |

- **验收结论**: **D0.5–D0.7 的判定仍悬置**（H-D0.5 未被有效检验）。
  **v2/v3 均发散 ⇒ 尚未产出可判定的 regime-extension 结果。**
- **产物**: `ckpt/follow_stage1_v3_speedext.zip`、`ckpt/follow_stage2_v3_speedext.zip`、
  `ckpt/probe_highonly_1m.zip`、`results/20260922_D0_regime_ext/{train_v3_stage1.log,train_v3_stage2.log,probe_highonly.log}`、
  `probe_highonly.py`
- **遗留 / 下一步**:
  1. **根因已定位为「高速段」**（见 §3 更正）⇒ 下一步是**针对高速段的条件**：
     实测 `v_cmd = clip(v_l + 0.8a, 0, 1.3)` 在 `v_l=1.0` 时**纠错权限仅 1.30×**
     （v=0.30 时 3.67×），且 P+FF 基线饱和率 0%→6%。
     **这是冻结先验的结构属性 ⇒ 若要改，触发铁律 2**（版本注 + sha256 + 重训）；
  2. **不建议**再调 `ent_coef`/LR 等超参——根因已定位为环境结构，超参搜索是绕路；
  3. **须与负责人裁决**：是否接受在**冻结文件**上改高速段控制权限（铁律 2 代价），
     或**收窄声明**（承认孪生仅在 v ≤ 0.50 可信）。**这是决策点，不是 agent 自作主张的范围。**

---

## [2026-09-22] H-D0.8 动作重参数化：两臂均未收敛 → 落定选项 2（收窄声明）

- **命令**:
  ```
  uv run python train_gain_sched.py --arm A     # g_high=0.4（收缩饱和质量）
  uv run python train_gain_sched.py --arm B     # g_high=1.6（放大残差，owner 原述方向）
  ```
- **更新次数（铁律 4）**: 各 610 ✅
- **前置闸门**: `step05_gain_gate.py` 四项全过（含**两次故障注入自检**：
  净化的增益调度、构造期抛异常的 worker —— 均被检出）。**闸门已自证会报警。**
- **冻结文件**: `follow_env.py` sha256 全程未变（wrapper 侧动作重标定）

### 结果：两臂均未收敛，H-D0.8 未通过

| 臂 | g_high | 末 std | 判定 |
|---|---|---|---|
| A | 0.4 | **4.45** | ❌（阈值 ≤2.0） |
| B | 1.6 | **5.17** | ❌ |

**六次 stage2 跑次全表**（同 610 updates）：

```
v1    U(0.25,0.50)            std 1.40  稳定
v2    U(0.25,1.00) g=1.0      std 6.13  发散
v3    U(0.25,1.00) 窄stage1   std 3.98  发散
probe U(0.80,1.00) 纯高速     std 4.40  发散
armA  U(0.25,1.00) g=0.4      std 4.45  发散
armB  U(0.25,1.00) g=1.6      std 5.17  发散
```

### 执行树判定：**第三支——机制证伪，落定选项 2**

- 动作重参数化**不能**挽救宽带训练（两方向均失败）；
- **选项 2（收窄声明）生效**：`docs/scenario_scope_statement_DRAFT.md`，
  「平台速度上界」入档（含 v2/v3/probe/armA/armB 五次发散证据）。

### 机制边界（重要，防过度结论）

`_check_reachable.py` 实测：`v_cmd` **可达集与增益无关**
（`[max(0,v_l−0.8), min(1.3,v_l+0.8)]`，由冻结文件内的 `V_MAX`/`ACT_GAIN` 锁定）。
⇒ **本实验从未能检验「权限」假设**。故：

- **被证伪**：「动作重参数化足以解决宽带发散」（更强的说法）；
- **未证实也未证伪**：「权限不足是根因」（需改冻结文件的 `V_MAX`/`ACT_GAIN`，触发铁律 2）。

- **验收结论**: H-D0.8/H-D0.8b **未通过**；执行树落第三支；
  选项 1（改冻结文件）**维持否决**（owner 已裁）；**选项 2 生效**。
- **产物**: `ckpt/follow_stage2_v4_gainA.zip`、`ckpt/follow_stage2_v4_gainB.zip`、
  `results/20260922_D0_regime_ext/{train_v4_armA.log,train_v4_armB.log,gain_gate.log}`、
  `gain_sched_env.py`、`train_gain_sched.py`
- **遗留 / 下一步**:
  - 选项 2 草案已就绪（`docs/scenario_scope_statement_DRAFT.md`），**待 owner 定稿**；
  - **Phase D 采购论证**依选项 2 限定：**Sim2Real 对照区间限于 v ≤ 0.50**，
    不得引用高速区结论；
  - 若日后要做平台级 env v2（放开 `V_MAX`/`ACT_GAIN`），须**先有因果证据**，
    成本为铁律 2 全套（版本注 + sha256 重铸 + 全部相关训练重来）。

---