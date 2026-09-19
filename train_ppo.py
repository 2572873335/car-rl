
"""
Week 2 - PPO training + ablation evaluation for FollowEnv
=========================================================
train :  uv run python train_ppo.py train --timesteps 500000 --n-envs 32
watch :  uv run tensorboard --logdir tb_logs
eval  :  uv run python train_ppo.py eval --model ckpt/best_model.zip
         (evaluates RL vs rule baselines on identical fixed seeds -> the
          ablation table for the report)
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from follow_env import FollowEnv, baseline_action, D_DES


def make_env(rank, easy=False):
    """Each parallel worker gets its own rng stream (task variation +
    domain randomization are resampled every episode inside the env).
    easy=True -> curriculum stage 1: short gaps, gentle leaders only."""
    def _init():
        env = FollowEnv(domain_randomize=True,
                        gap_max=0.5 if easy else 1.2,
                        behaviors=("constant", "sinusoid") if easy
                        else ("constant", "sinusoid", "brake"))
        env.reset(seed=10_000 + rank)
        return env
    return _init


# ------------------------------------------------------------------ train --
def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    os.makedirs("ckpt", exist_ok=True)
    venv = VecMonitor(SubprocVecEnv([make_env(i, args.easy)
                                     for i in range(args.n_envs)]))

    def _eval_env():
        e = FollowEnv(domain_randomize=False)
        e.reset(seed=999)             # fixed eval seeds -> comparable curves
        return e
    eval_env = DummyVecEnv([_eval_env])

    # 32 envs x 256 steps = 8192 steps per gradient update ->
    # --timesteps 5e6 gives ~610 updates (PPO needs a few hundred to converge)
    if args.load:
        model = PPO.load(args.load, env=venv,
                         tensorboard_log="./tb_logs", verbose=1)
        print(f"continuing from {args.load}")
    else:
        model = PPO("MlpPolicy", venv,
                    learning_rate=3e-4, n_steps=256, batch_size=512,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01,          # keep exploring: the residual
                                            # around the prior must be found
                    policy_kwargs=dict(net_arch=[128, 128]),
                    tensorboard_log="./tb_logs", verbose=1, seed=args.seed)

    cb = EvalCallback(eval_env, best_model_save_path="./ckpt",
                      log_path="./ckpt", eval_freq=5000,
                      n_eval_episodes=5, deterministic=True)
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model.save("ckpt/final_model")
    venv.close()
    print("done. best model -> ckpt/best_model.zip")


# --------------------------------------------------------------- evaluate --
def evaluate(policy, seeds, dr):
    """Roll out one policy over fixed seeds; every policy sees IDENTICAL
    leader behavior / initial gaps / randomization -> fair comparison."""
    rows, logs = [], []
    for s in seeds:
        env = FollowEnv(domain_randomize=dr)
        obs, _ = env.reset(seed=int(s))
        es = []
        while True:
            obs, _, term, trunc, _ = env.step(policy(obs))
            es.append(obs[0])                       # gap error [m]
            if term or trunc:
                break
        e = np.array(es)
        settled = np.abs(e[int(len(e) * 0.2):])     # skip catch-up transient
        rows.append(dict(seed=s, behavior=env.behavior, gap0=env.gap0,
                         steps=len(e), reason=env.term_reason,
                         mean_abs_e=settled.mean() * 100,
                         max_abs_e=np.abs(e).max() * 100))
        logs.append(env.log)
    return rows, logs


def eval_cmd(args):
    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    seeds = args.seeds if args.seeds else list(range(1000, 1010))
    print(f"evaluating {len(seeds)} fixed seeds (domain_randomize={args.domain_randomize})")

    policies = {
        "RL(PPO)":       lambda o: model.predict(o, deterministic=True)[0],
        "rule P":        lambda o: baseline_action(o, use_ff=False),
        "rule P+FF":     lambda o: baseline_action(o, use_ff=True),
    }
    all_logs = {}
    for name, pol in policies.items():
        rows, logs = evaluate(pol, seeds, args.domain_randomize)
        all_logs[name] = logs
        if args.verbose:
            for r in rows:
                print(f"    seed={r['seed']} {r['behavior']:9s} gap0={r['gap0']:.2f} "
                      f"steps={r['steps']:4d} {r['reason']:8s} "
                      f"settled|e|={r['mean_abs_e']:6.2f}cm")
        col = sum(r["reason"] == "collision" for r in rows)
        lost = sum(r["reason"] == "lost" for r in rows)
        off = sum(r["reason"] == "offtrack" for r in rows)
        print(f"{name:12s} mean|e|={np.mean([r['mean_abs_e'] for r in rows]):6.2f} cm  "
              f"worst|e|={np.max([r['max_abs_e'] for r in rows]):6.2f} cm  "
              f"collision={col} lost={lost} offtrack={off} /{len(rows)}")

    plot_compare(all_logs["RL(PPO)"][0], all_logs["rule P+FF"][0], args.out)


def plot_compare(log_rl, log_base, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(log_rl["t"], np.array(log_rl["gap"]) * 100, label="RL follower")
    ax1.plot(log_base["t"], np.array(log_base["gap"]) * 100, label="rule P+FF")
    ax1.axhline(D_DES * 100, color="r", ls="--", lw=1, label="target")
    ax1.set_ylabel("gap [cm]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title("gap (same leader, same seed)")
    ax2.plot(log_rl["t"], log_rl["v_l"], "k--", lw=1, label="leader v")
    ax2.plot(log_rl["t"], log_rl["v_f"], label="RL follower v")
    ax2.plot(log_base["t"], log_base["v_f"], label="rule follower v")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("v [m/s]")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"figure saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--timesteps", type=int, default=5_000_000)
    pt.add_argument("--n-envs", type=int, default=32)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--easy", action="store_true",
                    help="curriculum stage 1: short gap, gentle leaders")
    pt.add_argument("--load", default=None,
                    help="continue training from a checkpoint (stage 2)")
    pt.set_defaults(func=train)

    pe = sub.add_parser("eval")
    pe.add_argument("--model", default="ckpt/best_model.zip")
    pe.add_argument("--seeds", type=int, nargs="*", default=None)
    pe.add_argument("--domain-randomize", action="store_true")
    pe.add_argument("--out", default="eval_compare.png")
    pe.add_argument("-v", "--verbose", action="store_true")
    pe.set_defaults(func=eval_cmd)

    args = ap.parse_args()
    args.func(args)
