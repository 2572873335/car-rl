# 数据与版本管理规范（DATA_MANAGEMENT.md）

> 本文档定义本项目的目录、命名、归档与复现规范。**所有实验必须遵守**，
> 违反者视为返工（对应铁律 7：每修改一次环境或超参，RUNLOG.md 记一条，
> 旧文件归档不删除）。

## 1. 版本控制

- **仓库**：`/home/zy/car_rl/code0919`（git）。代码与文档进版本库。
- **忽略**（`.gitignore`）：`.venv/`、`__pycache__/`、`tb_logs*/`、`*.zip`、
  `*.npz`、仓库根 `*.png`、`*:Zone.Identifier`、编辑器目录。
- **模型与大数据**：用 **git tag 标记里程碑** + 目录归档，不直接进版本库
  （体积大、可再生）。`*.zip` 已忽略，物理存放于 `ckpt/` 与 `ckpt_ot/`。
- **里程碑 tag**：

  | tag | 指向 | 含义 |
  |---|---|---|
  | `milestone/phase0-env` | 3f2dfdc | 环境 + 健康检查通过 |
  | `milestone/phase1-follow` | 6503a13 | 跟车复刻完成（1.00cm 零碰撞） |
  | `milestone/phase2-overtake` | 897a07e | 超车复刻完成（final 10/10，1.4s） |

## 2. 目录规范

```
results/YYYYMMDD_<实验名>/
├── config.json      # seed, timesteps, n_envs, load_from, git commit, 超参
├── eval_*.txt       # 评估命令的**原始输出**（不可编辑、不可美化）
├── train_*.log      # 训练 stdout 原始日志
└── fig_*.png        # 曲线图/对比图（版本化命名，禁止覆盖旧图）
```

当前实例：

- `results/20260919_phase0_healthcheck/` — 环境健康检查
- `results/20260919_phase1_follow/` — 跟车两阶段训练 + 评估
- `results/20260919_phase2_overtake/` — 超车 BC 热身 + 微调 + 评估

## 3. checkpoint 命名

- 格式：`<任务>_<阶段>_v<N>.zip`，**禁止裸名覆盖**。
- 裸名 `best_model.zip` / `final_model.zip` 是训练脚本的固定输出（SB3 约定），
  训练结束后**立即**复制为带版本号的归档名：

  | 任务 | 归档 checkpoint |
  |---|---|
  | 跟车 stage1 | `ckpt/follow_stage1_v1.zip`（best）, `follow_stage1_final_v1.zip` |
  | 跟车 stage2 | `ckpt/follow_stage2_best_v1.zip`, `follow_stage2_final_v1.zip` |
  | 超车 BC | `ckpt_ot/overtake_bc_v1.zip` |
  | 超车微调 | `ckpt_ot/overtake_best_v1.zip`, `overtake_final_v1.zip` |

- **best/final 双测**（铁律 6）：每次评估必须同时对 best 与 final 出数，
  因为 best 的"最优"是按回合回报选的，可能被回合长度偏置（论文 5.7 节，
  本轮已复现：超车 best = 0/10 跟随策略，final = 10/10）。

## 4. 图表命名

- 格式：`fig_<任务>_<内容>_<版本>.png`，`--out` 显式版本化命名。
- 评估图禁止使用默认文件名（`eval_overtake.png` 等），否则互相覆盖（铁律 6、
  论文表 3 案例 14）。
- 报告引用图统一重命名为 `图N_<名称>_vN.png` 并在报告文末建立索引表（Phase 4）。

## 5. RUNLOG 规范

- 每轮实验一条记录：日期 / commit / 超参 / 命令 / **原始输出摘录**（铁律 1）/
  验收结论 / 与论文基准差异 / checkpoint / 归档路径 / 遗留 / 下一步。
- 旧条目**只增不改**；环境或超参一变即新增一条（铁律 7）。

## 6. 复现步骤（从零）

```bash
cd /home/zy/car_rl/code0919
uv venv --python 3.13 && uv pip install -r requirements.txt   # torch 走 cu128

# 健康检查（铁律 3，任何训练前必做）
uv run python car_following_sim.py                  # mean|e| <= 1.2cm
uv run python train_ot.py eval --rule-only          # overtake=10/10, collision=0

# Phase 1 跟车
uv run python train_ppo.py train --easy --timesteps 2500000 --n-envs 32 --seed 0
uv run python train_ppo.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt/best_model.zip
uv run python train_ppo.py eval --model ckpt/final_model.zip -v --out results/<date>_phase1_follow/fig_follow_eval_final_nominal.png
uv run python train_ppo.py eval --model ckpt/final_model.zip --domain-randomize -v --out results/<date>_phase1_follow/fig_follow_eval_final_dr.png

# Phase 2 超车
uv run python train_ot.py pretrain --n-demos 300 --bc-epochs 10
uv run python train_ot.py eval --model ckpt_ot/bc_model.zip -v
uv run python train_ot.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt_ot/bc_model.zip
uv run python train_ot.py eval --model ckpt_ot/final_model.zip -v --out results/<date>_phase2_overtake/fig_overtake_final_nominal.png
```

## 7. 铁律 4 合规核对

| 训练 | 命令 timesteps | 更新次数 = ts/(n_envs×n_steps) | ≥300 |
|---|---|---|---|
| 跟车 stage1 | 2,500,000 | 2.5M/(32×256) = **305** | ✅ |
| 跟车 stage2 | 5,000,000 | 5M/(32×256) = **610** | ✅ |
| 超车微调 | 5,000,000 | 5M/(32×256) = **610** | ✅ |

> 注：论文附录 A 的 stage1 命令为 2,000,000 步，仅产生 244 次更新，**不满足
> 铁律 4（≥300）**。本轮按铁律 4 上调至 2,500,000（305 次），其余命令不变。
