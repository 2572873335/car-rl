# demos_v1.npz — 规则演示数据集（RQ2 离线 RL 输入）

## 生成

```bash
uv run python export_demos.py --n-demos 300 --out demos_v1.npz
```

- **脚本**：`export_demos.py`（入库，numpy-only，不引入 d3rlpy）
- **采集策略**：`overtake_env.baseline_action_ot`（规则状态机），
  seed = `30000 + i`（与 `train_ot.py pretrain` 完全同 seed 族 → 与 BC 基线可比）
- **时间**：2026-09-20（Phase A 前置）

## 内容

| 字段 | shape | 说明 |
|---|---|---|
| `observations` | (41936, 6) | 归一化观测 |
| `actions` | (41936, 2) | [a_speed, a_lane]，规则机输出 |
| `rewards` | (41936,) | 逐步奖励（mean 0.55，成功 +60 含在内） |
| `next_observations` | (41936, 6) | 与 obs[t+1] 逐点吻合（已验证） |
| `terminals` | (41936,) | terminated \| truncated |
| `timeouts` | (41936,) | truncated（时间限/超时） |
| `episode_ends` | (300,) | 每集末尾下标 |
| `episode_reasons` | (300,) | 每集终止原因（全 success） |
| `episode_seeds` | (300,) | 每集 seed |
| `episode_terminals` | (41936,) | 真 MDP 终止（非 time-limit） |

- 集数 **300**，步数 **41936**（与论文"4.2 万"一致）
- 300/300 **success**（规则机在演示 seed 下从不失败，`--rule-only` 亦 10/10）

## 用途与边界

- **RQ2 输入**：`d3rlpy.MDPDataset(observations, actions, rewards, terminals, timeouts, episode_terminals)`。
- **BC 对照**：同数据训出 BC 2/10（论文 4.5 节），本数据集是 IQL/CQL 的同口径输入。
- **已知边界（positive-only）**：规则机必胜 → 数据集**只有成功轨迹**。
  离线 RL 由此只能学到"好行为分布"，**学不到失败规避**。
  这是特性而非缺陷——它使 RQ2 的"安全边界"维度有了对比基线；
  若要评估离线 RL 的失败行为，需另采含失败轨迹的数据（如 PPO 微调初期的 rollout）。

## 校验

```bash
uv run python -c "
import numpy as np
d = np.load('demos_v1.npz', allow_pickle=True)
assert d['observations'].shape == (41936, 6)
assert len(d['episode_ends']) == 300
assert (d['episode_reasons'] == 'success').all()
print('OK', d['observations'].shape)
"
```

## sha256（入 DATA_MANAGEMENT §8 数据资产）

```
4b31c47e7605df0eb269a02eeb46d8f6dbab97cf7b312e38d49045193c47f429  demos_v1.npz
```
