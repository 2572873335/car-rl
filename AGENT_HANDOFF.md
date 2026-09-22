# AGENT HANDOFF — car-rl 项目交接

> **给接手者**：你是一个全新的 agent，没有此前对话的上下文。本文档让你能在
> 15 分钟内接手这个项目。读完后，你应当能：跑通环境、理解当前进度、知道下一步做什么、
> 并遵守项目的执行纪律。
>
> **最后更新**：2026-09-20（Phase C W5 探针完成后）

---

## 0. 一分钟速览

**项目**：把 2022 年 TI 杯电赛 C 题（双小车跟随行驶）重构为强化学习问题。
在自建二维数字孪生中训练"跟车"（保持 20cm 间距）与"超车"（自主换内圈超越慢车）
两个策略，与手工规则严格对照。

**成绩**：跟车间距误差 1.00cm（规则最优 1.93cm）、零碰撞；超车 10/10 成功、
零失误、1.4s（规则 2.0s）。**独立复刻 11/11 指标一致**。

**现状**：Phase 0–4（复刻）+ Phase A（RQ2 离线 RL 研究）+ Phase B W3（开源发布）
已完成。仓库公开，Release v1.0.0 已上线。
**Phase C 已中止（RQ3 未决），D（Sim2Real）/ E（投稿）未启动。**

**Phase C 为何中止（2026-09-22）**：探针（W5，09-20）判明障碍是**非平稳性**而非机制；
但随后的 W6–M1 判据**连续四版被退化策略击穿**，且两次"更正"本身也被评审证伪
（`ERRATUM4_f23_correction_wrong.md`、`reviews/20260921_phaseC_m1_review1.md`）。
**在无法构建有效验收判据时 M1 无法开训，遂停。**

**仍成立的资产（勿丢弃）**：封堵者基准 `blocker@v1`（几何退化族在其下全 0.00）、
反制脚本完成率 0.77（可复现，证明题目可解）、真正的获胜条件
（封堵者物理位于外圈时切入）、以及 F14/F20/F23 的机制尸检（仓库最有价值的内容）。

**若要恢复 RQ3**：须先解决"如何测量对抗能力"——命中率类判据已被证伪
（随机行为命中率更高），可试评审建议的**因果判据**（消融时机能力后完成率
须显著下降，正例 Δ≈0.47）。详情见 `research/roadmap.md` Phase C 节。

**位置**：WSL Ubuntu，`/home/zy/car_rl/code0919`（git 仓库）。
远端 `git@github.com:2572873335/car-rl.git`（public）。

---

## 1. 环境与运行

```bash
# WSL 进入项目
wsl -e bash -lc "cd /home/zy/car_rl/code0919 && <cmd>"
```

**关键：`uv` 不在非登录 shell 的 PATH 里**，所有 `uv` 命令必须用 `bash -lc`
（登录 shell）或绝对路径 `/home/zy/.local/bin/uv`。

**两个 venv**（**不要合并，见 §5 铁律）**：
- `.venv/` —— 主环境：Python 3.13、torch 2.11+cu128、sb3 2.9.0、gymnasium **1.3.0**
- `.venv-d3rlpy/` —— 离线 RL 专用：d3rlpy 2.8.1、gymnasium **1.0.0**（隔离，因为
  d3rlpy 会 pin 降级 gymnasium，污染主环境就毁掉冻结存证）

**自检三步**（任何工作前先跑）：
```bash
wsl -e bash -lc "cd /home/zy/car_rl/code0919 && make check"   # 健康检查
wsl -e bash -lc "cd /home/zy/car_rl/code0919 && make demo"    # 全流程 smoke test (~5min)
```

---

## 2. 项目结构

```
# 冻结的环境（铁律 2，改则须重训 + 更新 sha256）
car_following_sim.py   数字孪生：赛道几何、差速运动学、纯追踪
follow_env.py          跟车 Gym 环境
overtake_env.py        超车 Gym 环境

# 训练与评估（主 venv）
train_ppo.py           跟车：PPO 训练 + 消融评估
train_ot.py            超车：BC 热身 + PPO 微调 + 评估
make_curves.py         从训练 stdout 解析曲线
robustness_sweep.py    执行器退化连续扫描（Phase A W1）
export_demos.py        导出演示数据集 demos_v1.npz
manual_bc_sweep.py     manual-BC 数据效率（H6）
manual_bc_h7.py        manual-BC 失败机制（H7）
plot_rq2_efficiency.py RQ2 数据效率图

# 离线 RL（.venv-d3rlpy）
rq2_offline.py         IQL/TD3+BC 训练与评估
final_eval_30.py       30-seed 终版评估（转录落盘）

# 文档
README.md              对外入口（AGV 双重叙事）
project_paper/         project_report_full.md（含 §4.8/§5.4.1 新章）+ 方法论文 + figures_v2/
RUNLOG.md              每次运行一条记录（只增不改）
DATA_MANAGEMENT.md     数据/版本规范 + §10 评审流程 + §8 代码哈希
research/roadmap.md    总路线图（Phase A–E）
research/ASSUMPTIONS.md 假设登记表（H1–H7，含事后修订披露）
results/YYYYMMDD_*/    原始输出、config、曲线（可溯源）
reviews/               评审记录（plan review + 报告 review）
```

---

## 3. 已完成的工作（按阶段）

| 阶段 | 内容 | 关键产物 |
|---|---|---|
| Phase 0–4 | 复刻论文基准 + 数据/版本规范 + 全量报告 | `project_report_full.md` §9–12 复刻验证 |
| Phase A W1 | 鲁棒性连续扫描（50 seed） | 发现 RL 失效边界（ε≥30% 稀有失效） |
| Phase A W2 | **RQ2 离线 RL**（IQL/TD3+BC/manual-BC/d3rlpy-BC/PPO 五方对照） | `rq2_offline.py`、§4.8 |
| Phase A A6 | 论文 v2 修订 | §4.8 成章、§5.4.1 扩展、摘要限定词 |
| Phase B W3 | 开源发布 | README/CITATION/Makefile、**Release v1.0.0** |

**核心研究结论（RQ2）**：
- **能力-效率谱系**：IQL 零交互即满分（离线数据足够支撑最优策略）；
  PPO 微调把 t_ot 从 2.0s 压到 1.4s（在线交互买效率）；
- **H6 成立**：manual-BC 瓶颈是**实现**而非数据（同 41936 条数据，d3rlpy-BC
  50 局满分，manual-BC 300 局仅 5/30）；
- **H7 证伪**：manual-BC 失败机制非损失函数、非学习率（多因素，future work）；
- **H1 证伪**：IQL 不是"介于 BC 与 PPO 之间"，而是追平在线微调。

---

## 4. 下一步（按 roadmap 优先级）

### 近期（Phase B W4，内容营销）
- **B5 博客**：①《"跟车"问题的 RL 解法：从零到 1.0cm》②《14 轮 debug 的工程纪律》
- **B6 渠道**：知乎/掘金 + V2EX + Reddit r/reinforcementlearning
- 这些需要**人工账号**发布；agent 可起草草稿。

### 中期（Phase C）——⛔ **已中止（2026-09-22），RQ3 未决**

**结论**：机制接通良好（SB3 自定义 `VecEnv` + 共享参数，`features_dim=6`），
但**完整联合自博弈不收敛**；**冻结一方即学会**。障碍是**非平稳性**。

**下一步（W6–W9）应从这个起点开始**：
1. **不要重跑完整联合自博弈探针**——已实测，勿重复；
2. 主路径：**冻结一方的联赛模式**——对手用快照，只训一方。
   这条路径**不需要**自定义 VecEnv、镜像观测、同生共死，
   可复用 `train_ot.py` 脚手架，工程成本远低于原计划；
3. W6 的**真正研究内容**：非平稳性缓解（对手池 / 快照历史 / 混合任务+对抗）；
> ⚠️ **【2026-09-21 撤回】** 本小节结论**已被证伪**，勿再引用。根因：作者适配器 `selfplay_to_frozen` 把 `gap` 硬编码为 1.0 m，**关掉了规则机的切入分支**（`baseline_action_ot` 仅在 `gap < 0.45` 时切入），故「零次用内圈」是**工具 bug 的伪影**。改正后规则机 **31/40 局用内圈**。详见 `ERRATUM_phaseC_findings.md`；保真度单测见 `results/20260920_phaseC_probe/scripts/adapter_fidelity_test.py`。

4. **重要警告**：冻结规则机 `baseline_action_ot` **不是博弈对手**
   （规则 vs 规则 40/40 局零次领先易手、零次用内圈）——
   不能拿它当陪练或胜率基线，须用**训练快照**。

原始输出：`results/20260920_phaseC_probe/`（3 份 raw + 11 脚本）；
自查发现：`findings_phaseC_selftest.md`；评审：`reviews/20260920_phaseC_probe_review1.md`。

### 远期
- **Phase D**：Sim2Real 真车（**硬件未下单**，需先采购 ~¥700；G3 决策门在 W12）
- **Phase E**：整合投稿（arXiv v3 / workshop）

---

## 5. 执行纪律（**铁律，违反即返工**）

来自 `DATA_MANAGEMENT.md`：

1. **数据真实性**：报告里每个数字必须来自真实命令输出，禁止编造/估算。
2. **环境冻结**：`follow_env.py`/`overtake_env.py` 冻结；改则须文件头注释版本号
   + 更新 sha256 表 + **相关训练从头重来**。
3. **健康检查先行**：任何训练前跑 `make check`（规则基线必须 10/10、零碰撞）。
4. **更新次数 ≥300**：`timesteps/(n_envs×n_steps) ≥ 300`，否则不许开训。
5. **终止原因全审计**：collision/lost/offtrack/success/failed 五类全报，
   只看成功率视为不合格。
6. **best/final 双测**：checkpoint 必须两个都测；图 `--out` 版本化命名，禁止覆盖。
7. **每轮一记**：环境或超参一变，RUNLOG 新增一条；旧文件归档**不删除**。

**新增纪律（`DATA_MANAGEMENT.md` §10 评审流程，2026-09-20 正式成文）**：
任何新方向/实验批次/环境改动/报告修订前，走
**plan → review → execute**：先写一页纸方案（假设/方法/验收/风险），交独立评审
逐条挑硬伤，意见存档 `reviews/`，处理完才动手。**假设必须训练前登记数值阈值**
（`research/ASSUMPTIONS.md`），落在阈值边界则补 seed 确认，不放宽阈值。

**评审流程已拦截 5 次真问题**（论文三轮 19 项 + sb3-contrib 库选型错误 +
报告 §4.8 的 30-seed 溯源缺失 + Phase C 探针的判据双假阴性）
——这是项目最值钱的流程资产。逐条清单见 `DATA_MANAGEMENT.md` §10.5。

---

## 5b. 文件落点纪律（**F19 通则**）

> **凡被 commit / 文档 / 勘误引用的文件，必须在仓库内有落点。**
> **禁止以 `/tmp` 作为其唯一居所。**

理由：`/tmp` 随重启蒸发（F18/F19 已两次因此失去证据链）；
而**被引用的东西找不到物证，引用（含撤回）的公信力即不成立**。

**配套**：`.gitignore` 会静默吞掉 `*.zip` / `*.npz`——新增资产须同步加例外
（见 `DATA_MANAGEMENT.md` §9）。废弃但被引用者入 `archive/<主题>_deprecated/` 并附 README。

## 6. 已知坑（前人血泪）

| 症状 | 根因 | 对策 |
|---|---|---|
| `uv: command not found` | 非登录 shell 无 PATH | 用 `bash -lc` |
| 脚本里 `$!`/反引号被吞 | Windows 侧 Git Bash 先展开 heredoc | **用 Write 工具写文件再拷**，别在 `wsl -e bash -c` 里塞含反引号的 heredoc |
| 30-seed 数字无出处 | 手写汇总冒充"原始输出" | 用脚本生成转录（`final_eval_30.py`），数字一律脚本产出 |
| d3rlpy 装上后主环境 gymnasium 被降级 | d3rlpy pin gymnasium==1.0.0 | **只装 `.venv-d3rlpy/`**，装完 `sha256sum -c` 验证冻结文件 |
| 离线扫描覆盖 ckpt | 同名 `.pt` 被覆写 | 用 `--tag` 区分；关键模型先归档 |
| `github.com` 连不上 | Windows 加速工具 hosts 污染到 127.0.0.1 | git 走 **SSH over 443**（`~/.ssh/config` 已配 `ssh.github.com:443`） |
| gh CLI 用不了 | 加速器自签证书 | 用 `curl -k` + PAT 调 API（release 创建/上传资产已验证可行） |
| 撞车后策略变保守 | "恐惧屏障"，正常现象 | 需 LfD 而非加罚（见论文 6.2） |
| best_model 反而是差策略 | 回合回报被回合长度偏置 | best/final 双测（论文 5.7，已复现） |
| SB3 报 `mat1 and mat2 shapes cannot be multiplied (2x4 and 8x64)` | SB3 **摊平 `observation_space` 前导维**（`(2,4)`→8 维），而每行只产出 4 维 | 声明**单 agent 形状** `(4,)`，用 `num_envs=2N` 表达多行；断言 `features_dim` |
| `SubprocVecEnv` 顶层实例化 → `EOFError: unexpected EOF` | forkserver 反复 re-import 本模块 | 训练脚本必须 `if __name__ == "__main__":` 守卫 |
| 基准数字与独立测量相反 | **并发训练进程**抢 CPU，污染计时 | **基准测试须在无其他训练时跑**；记录"机器空闲"状态 |
| 自博弈碰撞率反弹并钉死高位 | 非平稳性（MARL 常态） | 对手池/快照；或先退到"冻结一方"验证机制 |
| **验收闸门"子进程挂了仍打 PASS"** | 校验器自身没被验证 **（owner 2026-09-22）** | **任何验收闸门必须能用注入故障验证自己会报警**（kill 一个 worker 应得 FAIL）；校验器不 self-test 等于没有。例：`step05_gain_gate.py` 注入两个故障并断言被检出 |
| **采样分布大量质量被截断团成质点** | 未检查截断堆积 **（owner 2026-09-22）** | **任何采样分布先检查截断堆积**。例：`v_set ~ U(0.25,1.50)` 有 40% 区间落在 leader clip 之上，团成 1.0 质点 ⇒ 有效分布退化 |

---

## 7. 关键数字速查（论文/README 用）

```
跟车: RL 1.00cm(名义)/1.06cm(DR), 零碰撞 | P+FF 1.93cm | P 10/10 撞车
超车: RL 10/10 success, 零失误, 1.4s, 0.71m/s | 规则 2.0s, 0.55m/s
鲁棒性(50 seed): ε≤20% RL 50/50; ε=30% 47/50; 33% 41/50; 40% 37/50; 50% 27/50
                 P+FF 全程 50/50 零碰撞（但精度差 ~1.5×）
RQ2(30 seed): manual-BC 5/30 | d3rlpy-BC 30/30 | IQL 30/30 | TD3+BC 30/30(稳定版) | PPO 30/30
数据效率: IQL N=50 即 10/10; manual-BC N=300 仍 2/10
```

**权威出处**：`project_paper/project_report_full.md`；原始数据 `results/`；
每次运行 `RUNLOG.md`。

---

## 8. 接手第一步

```bash
# 1. 确认环境与提交状态
wsl -e bash -lc "cd /home/zy/car_rl/code0919 && git log --oneline -1 && git status -s"
# 2. 跑健康检查（必须全过）
wsl -e bash -lc "cd /home/zy/car_rl/code0919 && make check"
# 3. 读路线图决定方向
wsl -e bash -c "cd /home/zy/car_rl/code0919 && sed -n '95,145p' research/roadmap.md"
```

**若重启 Phase C（须先解决判据，见 §4）**：W5 探针**已完成**（见 §4）——**不要重跑完整联合自博弈探针**。
从**冻结一方的联赛模式**起步；W6 的研究内容是**非平稳性缓解**（对手池 / 快照历史 / 混合模式）。
仍按 §10 流程：先写 plan（`plan_phaseC_W6.md`）→ 独立评审 → 再动手。

**若只是问答/小改**：直接读 `project_report_full.md` 与 `RUNLOG.md` 即可。

---

## 9. 安全提醒

- 项目曾使用一个 GitHub PAT（已在会话中明文出现）——**若尚未撤销，应立即
  在 GitHub Settings → Developer settings 中删除**。
- 推送走 SSH key（`~/.ssh/id_ed25519`），无需 PAT。
