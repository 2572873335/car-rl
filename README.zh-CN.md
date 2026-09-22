> [English](README.md) | **简体中文**

<div align="center">

# car-rl

**学习跟随与超车**
面向低速车队机器人（AGV）的强化学习——固定间距跟随与自适应换道超车，
运行于可复现的数字孪生仿真中。

[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Reproduced](https://img.shields.io/badge/reproduction-11%2F11%20metrics%20matched-brightgreen)]()
[![Dataset](https://img.shields.io/badge/dataset-demos__v1-9cf)](demos_v1_README.md)

</div>

---

## 结果速览

| 任务 | 指标 | 本工作 | 最佳手工调参基线 |
|---|---|---|---|
| **跟随** | 稳态间距误差 | **1.00 cm**（域随机化下 1.06 cm） | 1.93 cm（P + 前馈） |
| | 碰撞次数 | **0** | 10/10（纯 P 控制） |
| **超车** | 成功率 | **10/10**，零失误 | 10/10（手工状态机） |
| | 平均超车耗时 | **1.4 s** | 2.0 s |

> **适用范围。** 上表中跟随任务的数字，是在训练所用的领车速度区间
> （0.25–0.50 m/s）内、按本仓库的统计口径测得的：取稳定窗口内的平均间距误差
> （剔除开局追赶瞬态），单位为观测单位（`200·|e|` m）。
> 一项场景保真度审计（D0）扫描了 11 个格子 × 3 个控制器 × 100 个种子，发现
> **精度优势仅限于训练速度带**：0.5 m/s 以下学习控制器在所有格子获胜，
> 0.55 m/s 边界处互有胜负，而**在 1.0 m/s 处带前馈的规则基线在所有格子获胜**
> （例如 0.50 m 间距：7.97 cm 对 5.73 cm；1.00 m 间距：7.65 cm 对 6.21 cm，
> 与上表同口径）。残差策略是**区间局部**的；结构先验（`v_l + k·e`）可以外推，
> 而学习到的残差在执行器摆率限制下已无余量。完整表格见
> [`docs/scenario_audit.md`](docs/scenario_audit.md)。把训练分布拓展到该区间
> 已被实测检验，**并无改善**：只要训练带包含该区间，训练就会失稳
> （见 *适用范围与已知边界*）。

**独立复刻**：对全流程的独立重跑，与论文的 **11/11** 项定量指标全部吻合
（见 [`project_paper/project_report_full.md`](project_paper/project_report_full.md) §9–12）。

---

## 这是什么

两辆低速车队机器人在同一条赛道上行驶，其中一辆（**跟随车**）需要：

1. **跟随**领车保持固定间距（20 cm），同时领车速度在变化
   （恒定 / 正弦 / 随机刹车）——即*跟车*问题；
2. **超车**一辆慢速领车：自主决定何时切入更短的内圈车道，再切回——
   即*自适应超车*问题。

两个控制器都用强化学习训练，并在完全相同的观测/动作接口下与手工规则对照。

> **两套叙事，同一套系统。** 用工业口径说，这是*面向低速 AGV 车队的
> 领车跟随与换道决策*（即上文采用的框架）。原始题目背景是 2022 年 TI 杯
> 电子设计竞赛的"小车跟随行驶"题，本项目将其重构为强化学习问题。
> 两种描述指向同一套代码、环境与结果。

## 为什么值得关注

- **分层架构**——*动作先验*（零动作 = 匹配领车速度）让安全性来自结构而非
  奖励博弈；强化学习只学残差。车辆动力学约束（曲率感知的速度上限）写在环境里，
  而不是留给策略去自行发现。
- **用 LfD 跨越深层探索谷**——超车任务存在"永远跟随"的局部最优，纯强化学习
  无法逃出；行为克隆热启动 + PPO 微调可以跨越（仅 BC 为 2/10；微调后为 10/10）。
- **能力–效率谱系**——离线强化学习（IQL）**无需任何环境交互**即可从演示数据
  达到满分成功，而在线微调买到的是速度（2.1 s → 1.4 s）。模仿学习的弱点被
  追溯到其*实现*，而非数据（见 §4.8）。
- **诚实的鲁棒性**——50 种子扫描显示：在每一个执行器退化档位上，强化学习都更准，
  但在训练分布内缘存在一条*失效边界*；规则基线从不失效，但精度差约 1.5 倍。
  同一教训也出现在速度轴上（见上文的适用范围注记）：
  **结构先验负责外推，学习到的残差是区间局部的**——这是对分层架构论证的一次细化。
- **去中心化是构造性的**——跟随决策由车端根据 V2V 广播的领车速度做出，
  没有中心调度器。在车队场景下这一点很重要：集中式车队管理（FMS/WCS）
  把通信与计算都集中起来，因此去中心化跟随是车队规模增长时的必要补充，
  而不只是风格选择。

## 适用范围与已知边界

本系统被验证能做什么、不能做什么，是**写下来**的，而不是暗示的。
三条边界，各自附出处：

1. **可信包络：v ≤ 0.50 m/s。** 所有头条跟随结果都在训练速度带内测得。
   超出该区间，学习控制器的精度优势不再成立。
   → [`docs/scenario_audit.md`](docs/scenario_audit.md)
2. **1.0 m/s 处优势反转。** 在所有测试格子上，带前馈的规则基线都比学习策略更准
   ——其差距是 0.55 m/s 边界处差距的 6 倍至 24 倍，而后者仅 0.086 cm
   （真值口径），实际可忽略。
   → [`docs/scenario_audit.md`](docs/scenario_audit.md) §2
3. **平台速度上界。** 只要训练带包含高速区间，训练就会失稳——表现为策略动作噪声
   发散，而不是性能渐变；且与频带宽度、课程设置、动作参数化方向均无关。
   → [`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md) §2.3

本项目的权威范围声明（声称什么、不声称什么）见
[`docs/scenario_scope_statement.md`](docs/scenario_scope_statement.md)。

## 复现

```bash
# 0. 环境（uv；Blackwell GPU 用 cu128 索引的 torch）
uv venv --python 3.13
uv pip install -r requirements.txt

# 1. 健康检查 —— 永远先跑这两条
uv run python car_following_sim.py          # 跟车基线：mean|e| <= 1.2 cm
uv run python train_ot.py eval --rule-only  # 超车基线：10/10，0 碰撞

# 2. 跟车任务（两阶段课程）
uv run python train_ppo.py train --easy --timesteps 2500000 --n-envs 32 --seed 0
uv run python train_ppo.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt/best_model.zip
uv run python train_ppo.py eval --model ckpt/final_model.zip -v --out results/fig_follow_eval.png

# 3. 超车任务（BC 热启动 + PPO 微调）
uv run python train_ot.py pretrain --n-demos 300 --bc-epochs 10
uv run python train_ot.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt_ot/bc_model.zip
uv run python train_ot.py eval --model ckpt_ot/final_model.zip -v --out results/fig_overtake_eval.png
```

完整指南见 [`project_paper/project_report_full.md`](project_paper/project_report_full.md) 附录 A。

## 仓库结构

```
car_following_sim.py     数字孪生：赛道几何、差速运动学、纯追踪
follow_env.py            跟车 Gym 环境（冻结；sha256 存证）
overtake_env.py          超车 Gym 环境（冻结；sha256 存证）
train_ppo.py             跟车：PPO 训练 + 消融评估
train_ot.py              超车：BC 热启动 + PPO 微调 + 消融
rq2_offline.py           离线强化学习（IQL / TD3+BC）实验，30 种子协议
export_demos.py          演示数据集导出器（demos_v1.npz）
robustness_sweep.py      执行器退化连续扫描
project_paper/           全量报告 + 方法论文 + 图
results/YYYYMMDD_*/      每次实验的原始输出、配置、曲线
RUNLOG.md                每次运行的实验日志
DATA_MANAGEMENT.md       数据/版本规范与评审纪律
```

## 数据集

`demos_v1.npz`——300 集 / 41,936 条转移的规则式超车演示，包含完整的
`(obs, action, reward, next_obs, terminal)` 元组，可用于离线强化学习。
格式与边界条件见 [`demos_v1_README.md`](demos_v1_README.md)。

```python
import numpy as np
d = np.load("demos_v1.npz", allow_pickle=True)
d["observations"].shape   # (41936, 6)
d["actions"].shape        # (41936, 2)
```

## 评估协议

这里的每一个数字都使用同一套协议，这正是重点：

- **固定种子集**——所有被比较的策略面对相同的初始间距、相同的领车脚本、
  相同的随机化。
- **终止原因全审计**——碰撞 / 丢失 / 出界 / 成功 / 失败全部报告，绝不只是成功率。
- **best *与* final 两个 checkpoint 都测**（按回合回报选出的"best"常常是退化的
  永远跟随策略——见报告 §5.7）。
- **域随机化**结果与名义结果并列报告。
- 所有"一致"的断言都由重跑背书，而非声称——见复刻附录。

## 引用

若使用本工作，见 [`CITATION.cff`](CITATION.cff)。

## 许可

MIT —— 见 [`LICENSE`](LICENSE)。

---

**本文件为译本，canonical 版本为 [`README.md`](README.md)，可能滞后。**
