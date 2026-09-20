"""final_eval_30.py - 30-seed final-table evaluation with full audit transcript.

Fixes review finding #1 (provenance): every number in table 4.5 must come from
a runnable, logged evaluation.  This script rolls out all methods over seeds
2000-2029 (nominal + DR), logging per-seed rows AND a summary.

Offline models need d3rlpy (separate venv); manual-BC needs sb3 (main venv).
Run each part with the right interpreter and concatenate:

  .venv-d3rlpy/bin/python final_eval_30.py offline >  <out>
  .venv/bin/python        final_eval_30.py manual  >> <out>
  uv run python           final_eval_30.py ppo     >> <out>
"""
import argparse
import numpy as np

SEEDS = list(range(2000, 2030))   # 30 fixed seeds


def _roll(step_fn, dr, seeds):
    """Roll out one episode per seed; return per-seed tuples."""
    from overtake_env import OvertakeEnv
    rows = []
    for s in seeds:
        env = OvertakeEnv(domain_randomize=dr)
        obs, _ = env.reset(seed=int(s))
        while True:
            obs, _, term, trunc, _ = env.step(step_fn(obs))
            if term or trunc:
                break
        lg = env.log
        t = np.array(lg["t"]); delta = np.array(lg["delta"])
        ahead = t[delta > 0.05]
        rows.append((s, env.term_reason, len(ahead) > 0,
                     ahead[0] if len(ahead) else float("nan"),
                     float(np.mean(lg["v_f"]))))
    return rows


def _summarize(label, dr, rows):
    n = len(rows)
    succ = sum(r[2] for r in rows)
    col = sum(r[1] == "collision" for r in rows)
    off = sum(r[1] == "offtrack" for r in rows)
    lost = sum(r[1] == "lost" for r in rows)
    fail = sum(r[1] == "failed" for r in rows)
    t_ot = [r[3] for r in rows if r[2]]
    mean_v = np.mean([r[4] for r in rows])
    print(f"RESULT {label:16s} dr={str(dr):5s}: success={succ}/{n} "
          f"collision={col} offtrack={off} lost={lost} failed={fail} "
          f"t_ot={np.mean(t_ot) if t_ot else float('nan'):.2f}s "
          f"mean_v={mean_v:.3f}")


def _log(label, dr, rows):
    for s, reason, ot, t_ot, mv in rows:
        print(f"  {label:16s} dr={str(dr):5s} seed={s} {reason:9s} "
              f"overtaken={ot} t_ot={t_ot:.1f} mean_v={mv:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["offline", "manual", "ppo"])
    args = ap.parse_args()

    if args.which == "offline":
        import d3rlpy
        items = [("BC(d3rlpy)", "ckpt_offline/bc_d300.pt"),
                 ("IQL", "ckpt_offline/iql_d300.pt"),
                 ("TD3+BC(lrhlf)", "ckpt_offline/td3bc_d300_lrhlf.pt")]
        for name, path in items:
            model = d3rlpy.load_learnable(path)
            step = lambda o, m=model: m.predict(o.reshape(1, -1))[0]
            for dr in (False, True):
                rows = _roll(step, dr, SEEDS)
                _log(name, dr, rows)
                _summarize(name, dr, rows)

    elif args.which == "manual":
        from stable_baselines3 import PPO
        for name, path in [("BC(manual)", "ckpt_offline/manualbc_d300.zip"),
                           ("BC(manual,MSE)", "ckpt_offline/manualbc_d300_mse.zip")]:
            m = PPO.load(path)
            step = lambda o, mm=m: mm.predict(o, deterministic=True)[0]
            for dr in (False, True):
                rows = _roll(step, dr, SEEDS)
                _log(name, dr, rows)
                _summarize(name, dr, rows)

    elif args.which == "ppo":
        from stable_baselines3 import PPO
        from overtake_env import baseline_action_ot
        m = PPO.load("ckpt_ot/overtake_final_v1.zip")
        for dr in (False, True):
            for name, step in [("RL(PPO)", lambda o: m.predict(o, deterministic=True)[0]),
                               ("rule-based", lambda o: baseline_action_ot(o))]:
                rows = _roll(step, dr, SEEDS)
                _log(name, dr, rows)
                _summarize(name, dr, rows)


if __name__ == "__main__":
    main()
