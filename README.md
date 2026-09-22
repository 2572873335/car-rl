<div align="center">

# car-rl

**Learning to Follow and Overtake**
Reinforcement learning for low-speed fleet robots — fixed-gap following and adaptive overtaking,
in a reproducible digital-twin simulation.

[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Reproduced](https://img.shields.io/badge/reproduction-11%2F11%20metrics%20matched-brightgreen)]()
[![Dataset](https://img.shields.io/badge/dataset-demos__v1-9cf)](demos_v1_README.md)

</div>

---

## Results at a glance

| Task | Metric | This work | Best hand-tuned baseline |
|---|---|---|---|
| **Following** | steady-state gap error | **1.00 cm** (1.06 cm under domain randomization) | 1.93 cm (P + feed-forward) |
| | collisions | **0** | 10/10 (P control) |
| **Overtaking** | success rate | **10/10**, zero failures | 10/10 (hand state-machine) |
| | mean time-to-overtake | **1.4 s** | 2.0 s |

> **Operating range.** The following-task numbers above are measured over the
> leader-speed range used for training (0.25–0.50 m/s), in the repo's
> reporting convention: mean gap error over the settled window (the opening
> catch-up transient excluded), in observation units (`200·|e|` m).
> A scenario-fidelity audit (D0) found that **this accuracy advantage does not
> extend to AGV-typical speeds**: at 1.0 m/s the learned controller is *less*
> accurate than the rule baseline with feed-forward across *every* cell tested
> (e.g. at 0.50 m gap: 7.98 cm vs 5.74 cm; 1.00 m gap: 7.64 cm vs 6.20 cm,
> same convention as the table above). The residual policy is regime-local; the
> structural prior (`v_l + k·e`) extrapolates where the learned residual runs
> out of headroom under actuator slew limits. See the
> [D0 scenario audit](plan_D0_scenario_audit_v3.md); the fix under test is to
> widen the training distribution to the target regime (D0.5–D0.7).

**Independently reproduced**: an independent re-run of the full pipeline matched
**11 of 11** quantitative metrics from the paper (see
[`project_paper/project_report_full.md`](project_paper/project_report_full.md), §9–12).

---

## What this is

Two low-speed fleet robots share a track. One (the **follower**) must:

1. **Follow** the leader at a fixed gap (20 cm) while the leader varies its speed
   (constant / sinusoidal / random braking) — the *car-following* problem; and
2. **Overtake** a slow leader by autonomously deciding when to switch to a
   shorter inner lane, then return — the *adaptive overtaking* problem.

Both controllers are learned with reinforcement learning, and are compared
against hand-tuned rules under an identical observation/action interface.

> **Two framings, one system.** In industrial terms this is *leader-following
> and lane-change decision-making for low-speed AGV fleets* (§ the framing used
> above). The original problem statement is the 2022 TI Cup electronics-design
> contest "car-following" task, which this project reformulates as an RL problem.
> Both descriptions refer to the same code, environments, and results.

## Why it is interesting

- **Layered architecture** — an *action prior* (zero action = match leader speed)
  makes safety structural rather than a reward gamble; RL learns only the residual.
  Vehicle dynamics constraints (curvature-aware speed limits) are written into the
  environment, not left for the policy to discover.
- **LfD for a deep exploration valley** — the overtaking task has a "follow forever"
  local optimum that pure RL cannot escape; behavior cloning warm-start + PPO
  fine-tuning crosses it (BC alone is 2/10; fine-tuned RL is 10/10).
- **A capability–efficiency spectrum** — offline RL (IQL) reaches full success
  from demonstrations *without any environment interaction*, while online
  fine-tuning is what buys speed (2.1 s → 1.4 s). Imitation's weakness is traced
  to its *implementation*, not its data (see §4.8).
- **Honest robustness** — a 50-seed sweep shows RL is more accurate at every
  degradation level but has a *failure boundary* just inside the training
  distribution; the rule baseline never fails but is ~1.5× less accurate. The
  same lesson appears on the speed axis (see the operating-range note above):
  **the structural prior carries extrapolation, the learned residual is
  regime-local** — a refinement of the layered-architecture argument.
- **Decentralised by construction** — following decisions are made on-vehicle
  from V2V-broadcast leader speed, with no central scheduler. In a fleet this
  matters: centralised fleet management (FMS/WCS) concentrates both
  communication and computation, so decentralised following is a necessary
  complement as fleet size grows, not merely a stylistic choice.

## Reproduce

```bash
# 0. environment (uv, torch from the cu128 index for Blackwell GPUs)
uv venv --python 3.13
uv pip install -r requirements.txt

# 1. health checks — ALWAYS run these first
uv run python car_following_sim.py          # following baseline: mean|e| <= 1.2 cm
uv run python train_ot.py eval --rule-only  # overtaking baseline: 10/10, 0 collision

# 2. following task (two-stage curriculum)
uv run python train_ppo.py train --easy --timesteps 2500000 --n-envs 32 --seed 0
uv run python train_ppo.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt/best_model.zip
uv run python train_ppo.py eval --model ckpt/final_model.zip -v --out results/fig_follow_eval.png

# 3. overtaking task (BC warm-start + PPO fine-tune)
uv run python train_ot.py pretrain --n-demos 300 --bc-epochs 10
uv run python train_ot.py train --timesteps 5000000 --n-envs 32 --seed 0 --load ckpt_ot/bc_model.zip
uv run python train_ot.py eval --model ckpt_ot/final_model.zip -v --out results/fig_overtake_eval.png
```

See [`project_paper/project_report_full.md`](project_paper/project_report_full.md)
appendix A for the full guide.

## Repository layout

```
car_following_sim.py     digital twin: track geometry, differential-drive kinematics, pure-pursuit
follow_env.py            following Gym env (frozen; sha256-pinned)
overtake_env.py          overtaking Gym env (frozen; sha256-pinned)
train_ppo.py             following: PPO training + ablation evaluation
train_ot.py              overtaking: BC warm-start + PPO fine-tune + ablation
rq2_offline.py           offline RL (IQL / TD3+BC) experiments, 30-seed protocol
export_demos.py          demonstration-dataset exporter (demos_v1.npz)
robustness_sweep.py      continuous actuator-degradation sweep
project_paper/           full report + method paper + figures
results/YYYYMMDD_*/      per-experiment raw outputs, configs, curves
RUNLOG.md                per-run experiment log
DATA_MANAGEMENT.md       versioning + review discipline
```

## Dataset

`demos_v1.npz` — 300 episodes / 41,936 transitions of rule-based overtaking
demonstrations, with full `(obs, action, reward, next_obs, terminal)` tuples for
offline RL. Format and boundary conditions: [`demos_v1_README.md`](demos_v1_README.md).

```python
import numpy as np
d = np.load("demos_v1.npz", allow_pickle=True)
d["observations"].shape   # (41936, 6)
d["actions"].shape        # (41936, 2)
```

## Evaluation protocol

Every number here uses an identical protocol, which is the point:

- **Fixed seed sets** — same initial gaps, same leader scripts, same
  randomization across all compared policies.
- **Full termination audit** — collision / lost / offtrack / success / failed are
  all reported, never just success rate.
- **Best *and* final checkpoints** are both evaluated (the "best" by episode
  return is often a degenerate follow-forever policy — see report §5.7).
- **Domain randomization** results are reported alongside nominal ones.
- Reports of "identical" are backed by re-runs, not by assertion — see the
  reproduction appendix.

## Citation

If you use this work, see [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
