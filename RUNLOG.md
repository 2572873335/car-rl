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
