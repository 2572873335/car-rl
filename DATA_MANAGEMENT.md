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

## 8. 代码文件哈希（环境冻结存证）

以下 sha256 于 2026-09-19 复刻完成后在实际文件上计算（`sha256sum`），用于证明
**本轮全部训练/评估所用代码与论文最终版逐字节一致**（铁律 2：环境冻结）。
任何后续修改都会使哈希改变，届时按铁律 2 须在文件头注释版本号并**从头重训**。

| 文件 | sha256 |
|---|---|
| `car_following_sim.py` | `bfb1678d7a5bb10c60632b54ee77d3d357ecce6b21a4c3f0384ecca9824bec89` |
| `follow_env.py` | `941c447b969c2fa1f153ed57219235bc028f5ffa9e4052d341f92c97f91e4e2e` |
| `overtake_env.py` | `78ee12dd9d5cc6b874f5f530cd6808c7d8895d1aa6677a79559fab1423526b98` |
| `train_ppo.py` | `70136c0b761949662bba9ae8d4ceb246a390b7533a86bdf4cb5f21059aef38b9` |
| `train_ot.py` | `8197daa0c20d469fe8b3aae48466e54f59c0790b647f3b6d8cd933aa0c28f4d1` |

**核验命令**：

```bash
sha256sum -c <<'EOF'
bfb1678d7a5bb10c60632b54ee77d3d357ecce6b21a4c3f0384ecca9824bec89  car_following_sim.py
941c447b969c2fa1f153ed57219235bc028f5ffa9e4052d341f92c97f91e4e2e  follow_env.py
78ee12dd9d5cc6b874f5f530cd6808c7d8895d1aa6677a79559fab1423526b98  overtake_env.py
70136c0b761949662bba9ae8d4ceb246a390b7533a86bdf4cb5f21059aef38b9  train_ppo.py
8197daa0c20d469fe8b3aae48466e54f59c0790b647f3b6d8cd933aa0c28f4d1  train_ot.py
EOF
```

> 复刻开训前亦可用同一命令核对，确保所用代码未漂移（对应论文表 3 案例 8、
> 12："沙箱与用户结果矛盾 = 文件版本不一致" / "环境持续漂移"）。

## 9. 数据资产哈希

发布用数据集，同样以 sha256 存证（生成脚本 `export_demos.py` 入库）。

| 文件 | 说明 | sha256 |
|---|---|---|
| `demos_v1.npz` | 规则演示数据集（300 集 / 41936 转移），RQ2 离线 RL 输入 | `4b31c47e7605df0eb269a02eeb46d8f6dbab97cf7b312e38d49045193c47f429` |

- 生成：`uv run python export_demos.py --n-demos 300 --out demos_v1.npz`
- 说明文档：`demos_v1_README.md`
- **注意**：`.gitignore` 的 `*.npz` 规则会静默吞掉该文件，已加 `!demos_v1.npz`
  例外（否则 `git add` 不报错但文件不入库）。新增数据资产时需同步加例外。


## 10. 评审流程（plan → review → execute）

> **本章补写于 2026-09-20**。此前 `AGENT_HANDOFF.md`、`plan_W2_offline_rl.md`、
> `plan_A1_A2.md`、`RUNLOG.md`、`research/ASSUMPTIONS.md` 与 `reviews/` 下的
> 三份评审**共 12+ 处引用"§10 评审流程"**，但该章节**从未写入**（文件止于 §9）。
> 流程本身一直是真实执行的，此处把它正式成文——由 Phase C 探针的
> 独立评审与博客起草任务各自发现该缺口。

### 10.1 触发条件

**任何**下列动作**之前**，必须先走完整流程：

- 新研究方向、新实验批次；
- 环境（`follow_env.py`/`overtake_env.py` 等冻结文件）改动；
- 报告/论文的修订；
- 任何会写进结论的数字的产生方式变更。

### 10.2 三步流程

| 步 | 产出 | 要求 |
|---|---|---|
| **plan** | 一页纸方案 | 必须含：**假设 / 方法 / 验收阈值 / 风险与预案** 四节 |
| **review** | `reviews/YYYYMMDD_<主题>_reviewN.md` | **独立评审**（无作者上下文），逐条挑硬伤，给出明确裁决：Approve / Approve with amendments / Reject |
| **execute** | 实验 + `RUNLOG` 一条 | **评审意见处理完毕后才动手**；处理对照表附在 plan 末尾 |

### 10.3 假设预注册（**硬性**）

**任何训练/实验开跑前**，在 `research/ASSUMPTIONS.md` 登记假设与**判定阈值**：

- 阈值**训练前定死**；
- 结果落在阈值边界 → **补 seed 确认**，**不放宽阈值**；
- 若确需事后修订阈值，**必须显式披露**为 post-hoc 修订并说明理由
  （见 `ASSUMPTIONS.md` 的 H4 修订说明，作为诚信披露的范例）。

### 10.4 判据前置验证（**反例先行**）

> 沉淀于 Phase C：**四版判据**（W5 的胜率、W6 v1 的 [35%,65%] 带、
> Step 0-A 的碰撞率、Step 0-C 的非对称胜率）**全部可被退化策略通过**，
> 若未拦下会导致训练跑完却无法判定。详见 `step0_criteria_findings.md`。

**任何判据（含假设登记表中的阈值）在投入使用前，必须附"退化反例表"**：
列出**至少两个已知劣质行为**（不动、绕圈、乱动……）及其在该判据下的取值，
**证明判据能排除它们**。判据无法区分退化行为 → **该判据作废**，不得写入验收。

**构造性陷阱（F14）**：当**同一策略**同时驱动博弈双方时，
对局结果**完全由初始条件决定**——任何建立在"终局相对位置"上的指标
（胜率、终局差）**在构造上无法区分策略优劣**。
故自博弈的**验收判据**必须要么**消除初始条件变量**（固定落后起跑），
要么改用**任务级判据**（沿用环境自身的成功定义）。
"策略 vs 自身"类指标**只能作训练健康诊断，不得作验收**。

### 10.5 诚实条款

- 报告里每个数字必须来自**真实命令输出**，禁止编造/估算（铁律 1）；
- 假设的**证伪也是结果**，须如实登记（H1、H7 即为证伪记录）；
- 指标解读口径**不得事后随意更换**（如"平均回报无效"这类断言须有实测支撑）。

### 10.6 流程资产与历史拦截

**评审流程已五次在动手前拦截真问题**，这是项目最值钱的流程资产：

| # | 拦截内容 | 出处 |
|---|---|---|
| 1 | `sb3-contrib` 不提供离线 RL（库选型错误） | W2 评审 |
| 2 | W2 plan 数据映射描述错误（npz 字段语义不对） | `reviews/20260920_w2_plan_review1.md` R1 |
| 3 | d3rlpy 会 pin 降级 gymnasium，威胁冻结环境 | 同上 |
| 4 | 报告 §4.8 的 30-seed 溯源缺失 | `reviews/20260920_report48_review1.md` |
| 5 | Phase C 探针：回报信号误判 + 熵判据方向相反（双假阴性将错误枪毙 RQ3） | `reviews/20260920_phaseC_probe_review1.md` |

> **计数口径说明**：`AGENT_HANDOFF.md` §5 早前记"拦截 4 次"且括号内只列 3 项
> （数目与清单不自洽）。本表以**具名逐条**为准，截止 2026-09-20 为 **5 次**。

### 10.7 相关纪律（与铁律的关系）

本章流程**叠加**在铁律之上，不替代铁律。特别注意两条交叉：

- **铁律 2（环境冻结）**：plan 中若涉及冻结文件改动，评审须核对 sha256 表更新计划；
- **铁律 4（更新次数 ≥300）**：其语境是"**产出结论的训练**"。
  **冒烟测试与机制探针**（如 `make demo`、Phase C W5 探针）
  **形式上豁免**，但须在 plan 与 RUNLOG 中**显式声明豁免**并说明理由。
