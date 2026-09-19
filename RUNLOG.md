# RUNLOG — 复刻实验记录

> 规范：每轮实验一条记录（日期 / git commit / 超参 / 曲线与结果路径 / 验收结论 /
> 与论文基准的差异 / 备注）。环境与超参每修改一次即新增一条，旧条目不改写。
> 铁律：数据真实性（数字均来自真实命令输出）、环境冻结、健康检查先行、
> 更新次数≥300、终止原因全审计、checkpoint best/final 双测、图版本化命名。

---

## [2026-09-19] Phase 0 — 环境初始化与健康检查

- **commit**: （首次提交，见 git log）
- **目的**: 建 uv 环境、装依赖、跑通三项健康检查（铁律 3 前置）
- **环境**: Python 3.13.15 / uv 0.12.16 / torch 2.11.0+cu128 / sb3 2.9.0 /
  gymnasium 1.3.0 / numpy 2.5.2 / RTX 5060 8GB (driver 591.74, CUDA 13.1)
- **命令与原始输出**:

  ```
  $ uv run python car_following_sim.py
  == sim done: T=45.0s  v_set=0.3m/s  d_des=0.2m ==
    mean|gap err| = 1.10 cm
    max |gap err| = 2.50 cm
    last gap      = 18.78 cm
    follower lateral RMS = 2.66 cm
    near-collision steps (gap<12cm) = 0

  $ uv run python train_ot.py eval --rule-only
  evaluating 10 fixed seeds (domain_randomize=False)
  rule-based   overtake=10/10  collision=0  offtrack=0  lost=0  t_overtake=  2.0s  mean_v=0.55 m/s
  ```

- **验收结论**:
  - 跟车健康检查 mean|e| = 1.10 cm ≤ 1.2 cm ✅（无碰撞）
  - 超车规则基线 overtake=10/10, collision=0, offtrack=0 ✅
- **与论文基准差异**: 超车规则基线 2.0s / 0.55 m/s 与论文表 2 完全一致；
  跟车 sim 基线（纯 P，1.10cm）为 Week1 演示脚本，非表 1 的规则 P（撞车）口径。
- **原始输出归档**: results/20260919_phase0_healthcheck/
- **遗留问题**: 无
- **下一步**: Phase 1 跟车两阶段训练

---

## [2026-09-19] Phase 1 — 跟车任务复刻（两阶段课程）

- **commit**: 3f2dfdc（训练前基线）
- **超参**: PPO MlpPolicy[128,128], lr 3e-4, n_steps 256, batch 512, n_envs 32,
  γ 0.99, GAE λ 0.95, clip 0.2, ent_coef 0.01, seed 0
- **更新次数（铁律 4）**: stage1 2.5M/(32×256) = 305 ✅；stage2 5M/(32×256) = 610 ✅
  （论文原命令 stage1=2M → 仅 244 次，不足 300，已按铁律 4 上调至 2.5M）
- **命令**:

  ```
  # stage1 (easy 课程)
  uv run python train_ppo.py train --easy --timesteps 2500000 --n-envs 32 --seed 0
  # stage2 (完整任务，从 stage1 best 续训)
  uv run python train_ppo.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt/best_model.zip
  # 评估（名义 + 域随机化；best/final 双测，铁律 6）
  uv run python train_ppo.py eval --model ckpt/best_model.zip -v --out results/20260919_phase1_follow/fig_follow_eval_best_nominal.png
  uv run python train_ppo.py eval --model ckpt/best_model.zip --domain-randomize -v --out results/20260919_phase1_follow/fig_follow_eval_best_dr.png
  ```

- **训练曲线原始特征**（make_curves.py 解析 stdout）:
  - stage1: ep_len_mean 145 → 1000；ep_rew_mean −729 → −88（306 点）
  - stage2: ep_len_mean 94 → 974；ep_rew_mean −2650 → −910（611 点）
- **评估原始输出**（固定 10 seed，全终止原因审计，铁律 5）:

  ```
  == 名义参数 (domain_randomize=False) ==
  RL(PPO)      mean|e|=  1.00 cm  worst|e|=150.74 cm  collision=0 lost=0 offtrack=0 /10
  rule P       mean|e|= 37.98 cm  worst|e|=150.74 cm  collision=10 lost=0 offtrack=0 /10
  rule P+FF    mean|e|=  1.93 cm  worst|e|=150.74 cm  collision=0 lost=0 offtrack=0 /10

  == 域随机化 (domain_randomize=True) ==
  RL(PPO)      mean|e|=  1.06 cm  worst|e|=172.89 cm  collision=0 lost=0 offtrack=0 /10
  rule P       mean|e|= 43.55 cm  worst|e|=172.89 cm  collision=10 lost=0 offtrack=0 /10
  rule P+FF    mean|e|=  1.94 cm  worst|e|=172.89 cm  collision=0 lost=0 offtrack=0 /10

  == final_model 双测 (铁律 6) ==
  final 名义: RL mean|e|= 0.93 cm  collision=0
  final DR  : RL mean|e|= 0.91 cm  collision=0
  ```

- **验收结论**:
  - RL mean|e| = 1.00 cm ≤ 1.3 cm ✅ 且零碰撞 ✅
  - 规则 P 复现 10/10 撞车 ✅；P+FF ≈ 1.93 cm ✅
- **与论文基准差异**: 逐项完全一致（表 1：RL 1.00/1.06、P 37.98/43.55 撞车、
  P+FF 1.93/1.94）。final_model 略优于 best（0.93/0.91 vs 1.00/1.06）。
- **checkpoint**: ckpt/follow_stage1_v1.zip, follow_stage1_final_v1.zip,
  follow_stage2_best_v1.zip, follow_stage2_final_v1.zip（裸名 best/final 为最新）
- **原始输出归档**: results/20260919_phase1_follow/
- **遗留问题**: 无
- **下一步**: Phase 2 超车任务（BC 300 demos + 微调）

---
