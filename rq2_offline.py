"""rq2_offline.py - Phase A W2: RQ2 offline-RL experiments (IQL / TD3+BC).

Runs INSIDE the .venv-d3rlpy interpreter (gymnasium 1.0.0), NOT the main venv.
Reads demos_v1.npz read-only.  Only *evaluation* constructs an env; training
never touches it (RQ2's "offline" claim depends on this).

Data mapping (review1 R1, goal-as-terminal):
    terminals[end] = reason in {collision, offtrack, lost, success}
    timeouts[end]  = reason in {timeout, failed}
    (reconstructed from episode_reasons -- the npz's own terminals/timeouts
     fields are NOT usable: the env reports success as truncated=True.)

Usage:
    python rq2_offline.py gate            # 100-transition API pilot gate
    python rq2_offline.py train --algo iql --demos 300
    python rq2_offline.py eval  --algo iql --demos 300 [--domain-randomize]
    python rq2_offline.py sweep           # data-efficiency: 50/100/200/300
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

DEMOS = "demos_v1.npz"
CKPT_DIR = "ckpt_offline"
TRUE_TERM = {"collision", "offtrack", "lost", "success"}
EVAL_SEEDS = list(range(2000, 2010))   # same protocol as paper / W1


# ---------------------------------------------------------------- dataset --
def load_demos(path=DEMOS, max_episodes=None):
    """Build (observations, actions, rewards, terminals, timeouts) per mapping A.
    max_episodes: keep only the first N episodes (data-efficiency sweep)."""
    d = np.load(path, allow_pickle=True)
    obs, act, rew = d["observations"], d["actions"], d["rewards"]
    ends, reasons = d["episode_ends"], d["episode_reasons"]
    n = len(rew)

    if max_episodes is not None:
        last = int(ends[max_episodes - 1])
        obs, act, rew = obs[: last + 1], act[: last + 1], rew[: last + 1]
        ends, reasons = ends[:max_episodes], reasons[:max_episodes]
        n = last + 1

    terminals = np.zeros(n, dtype=np.float32)
    timeouts = np.zeros(n, dtype=np.float32)
    for end, reason in zip(ends, reasons):
        if str(reason) in TRUE_TERM:
            terminals[int(end)] = 1.0
        else:
            timeouts[int(end)] = 1.0
    return obs, act.astype(np.float32), rew.astype(np.float32), terminals, timeouts


def make_dataset(obs, act, rew, terminals, timeouts):
    from d3rlpy.dataset import MDPDataset
    return MDPDataset(observations=obs, actions=act, rewards=rew,
                      terminals=terminals, timeouts=timeouts)


# ---------------------------------------------------------------- algos ----
def make_algo(name, device="cpu"):
    from d3rlpy.algos import IQLConfig, TD3PlusBCConfig
    if name == "iql":
        return IQLConfig().create(device=device)
    if name in ("td3bc", "td3+bc"):
        return TD3PlusBCConfig().create(device=device)
    raise ValueError(f"unknown algo {name}")


# ------------------------------------------------------------------ gate ---
def gate():
    """100+-transition API pilot gate (review1: validate d3rlpy 2.x API before
    burning full machine time).

    NOTE: must slice on WHOLE episodes -- a raw obs[:100] slice lands mid-episode
    with no terminal/timeout flag, and MDPDataset rejects it
    ("No episode termination was found")."""
    print("=== GATE 1: whole-episode API pilot ===")
    obs, act, rew, term, tout = load_demos()
    ds = make_dataset(obs, act, rew, term, tout)   # full 300 episodes first
    n_trans = int(term.sum() + tout.sum())
    print(f"MDPDataset built from {len(obs)} transitions, "
          f"{len(ds.episodes)} episodes, {n_trans} terminal/timeout markers")
    # pilot fits on a small but WHOLE first-K slice
    k = 5
    sub = make_dataset(*load_demos(max_episodes=k))
    print(f"pilot slice: first {k} episodes, "
          f"{sub.transition_count} transitions")
    for name in ("iql", "td3bc"):
        algo = make_algo(name)
        algo.fit(sub, n_steps=100)
        acts = algo.predict(obs[:5])
        print(f"  {name:6s} fit OK; predict shape={np.asarray(acts).shape}")
    print("GATE 1 PASSED")


# ----------------------------------------------------------------- train ---
def train(algo_name, n_episodes, n_steps, device="cpu"):
    obs, act, rew, term, tout = load_demos(max_episodes=n_episodes)
    ds = make_dataset(obs, act, rew, term, tout)
    print(f"train {algo_name}: {n_episodes} episodes, {len(obs)} transitions, "
          f"n_steps={n_steps}")
    algo = make_algo(algo_name, device=device)
    algo.fit(ds, n_steps=n_steps)
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{algo_name}_d{n_episodes}.pt")
    algo.save(path)
    print(f"saved -> {path}")
    return path


def load_algo(algo_name, n_episodes, device="cpu"):
    """Load a saved offline model. d3rlpy 2.x: models have no .load() classmethod;
    the top-level d3rlpy.load_learnable rehydrates config+weights."""
    import d3rlpy
    path = os.path.join(CKPT_DIR, f"{algo_name}_d{n_episodes}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; train first")
    return d3rlpy.load_learnable(path, device=device)


# ------------------------------------------------------------------ eval ---
def eval_algo(algo_name, n_episodes, seeds=EVAL_SEEDS, dr=False, device="cpu"):
    """Roll out the offline policy in the env. This is the ONLY place that
    touches the env -- training never does (RQ2 offline claim)."""
    from overtake_env import OvertakeEnv
    algo = load_algo(algo_name, n_episodes, device=device)

    rows = []
    for s in seeds:
        env = OvertakeEnv(domain_randomize=dr)
        obs, _ = env.reset(seed=int(s))
        while True:
            a = algo.predict(obs.reshape(1, -1))[0]
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        lg = env.log
        t = np.array(lg["t"]); delta = np.array(lg["delta"])
        ahead = t[delta > 0.05]
        rows.append(dict(seed=s, reason=env.term_reason,
                         overtaken=len(ahead) > 0,
                         t_ot=ahead[0] if len(ahead) else np.nan,
                         mean_v=float(np.mean(lg["v_f"]))))
    return rows


def summarize_eval(rows):
    n = len(rows)
    succ = sum(r["overtaken"] for r in rows)
    col = sum(r["reason"] == "collision" for r in rows)
    off = sum(r["reason"] == "offtrack" for r in rows)
    lost = sum(r["reason"] == "lost" for r in rows)
    fail = sum(r["reason"] == "failed" for r in rows)
    t_ot = [r["t_ot"] for r in rows if r["overtaken"]]
    return dict(success=f"{succ}/{n}", n_succ=succ, n=n, collision=col,
                offtrack=off, lost=lost, failed=fail,
                t_ot=np.mean(t_ot) if t_ot else float("nan"),
                mean_v=np.mean([r["mean_v"] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gate").set_defaults(func=lambda a: gate())

    pt = sub.add_parser("train")
    pt.add_argument("--algo", required=True)
    pt.add_argument("--demos", type=int, required=True)
    pt.add_argument("--n-steps", type=int, default=100_000)
    pt.add_argument("--device", default="cpu")
    pt.set_defaults(func=lambda a: train(a.algo, a.demos, a.n_steps, a.device))

    pe = sub.add_parser("eval")
    pe.add_argument("--algo", required=True)
    pe.add_argument("--demos", type=int, required=True)
    pe.add_argument("--domain-randomize", action="store_true")
    pe.add_argument("--device", default="cpu")
    def _eval(a):
        rows = eval_algo(a.algo, a.demos, dr=a.domain_randomize, device=a.device)
        s = summarize_eval(rows)
        print(f"{a.algo:6s} d={a.demos:3d} dr={a.domain_randomize}: "
              f"success={s['success']} collision={s['collision']} "
              f"offtrack={s['offtrack']} lost={s['lost']} failed={s['failed']} "
              f"t_ot={s['t_ot']:.1f}s mean_v={s['mean_v']:.2f}")
    pe.set_defaults(func=_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
