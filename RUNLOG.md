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
