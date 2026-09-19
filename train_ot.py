
"""
Week 3 - PPO training + evaluation for the OvertakeEnv
======================================================
train :  uv run python train_ot.py train --timesteps 5000000 --n-envs 32
eval  :  uv run python train_ot.py eval --model ckpt_ot/best_model.zip
         metrics: overtake success rate, time-to-overtake, collisions,
                  mean follower speed  (RL vs hand-crafted state machine)
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from overtake_env import OvertakeEnv, baseline_action_ot


def make_env(rank, easy=False):
    def _init():
        env = OvertakeEnv(domain_randomize=True,
                          gap_max=0.5 if easy else 1.0,
                          behaviors=("constant",) if easy
                          else ("constant", "sinusoid", "brake"))
        env.reset(seed=20_000 + rank)
        return env
    return _init


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    os.makedirs("ckpt_ot", exist_ok=True)
    venv = VecMonitor(SubprocVecEnv([make_env(i, args.easy)
                                     for i in range(args.n_envs)]))

    def _eval_env():
        e = OvertakeEnv(domain_randomize=False)
        e.reset(seed=888)
        return e
    eval_env = DummyVecEnv([_eval_env])

    if args.load:
        model = PPO.load(args.load, env=venv, tensorboard_log="./tb_logs_ot", verbose=1)
        print(f"continuing from {args.load}")
    else:
        model = PPO("MlpPolicy", venv,
                    learning_rate=3e-4, n_steps=256, batch_size=512,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    tensorboard_log="./tb_logs_ot", verbose=1, seed=args.seed)
    cb = EvalCallback(eval_env, best_model_save_path="./ckpt_ot",
                      log_path="./ckpt_ot", eval_freq=5000,
                      n_eval_episodes=5, deterministic=True)
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model.save("ckpt_ot/final_model")
    venv.close()
    print("done. best model -> ckpt_ot/best_model.zip")


def rollout(policy, seeds, dr):
    rows, logs = [], []
    for s in seeds:
        env = OvertakeEnv(domain_randomize=dr)
        obs, _ = env.reset(seed=int(s))
        while True:
            obs, _, term, trunc, _ = env.step(policy(obs))
            if term or trunc:
                break
        lg = env.log
        t_arr = np.array(lg["t"]); delta = np.array(lg["delta"])
        ahead_t = t_arr[delta > 0.05]
        overtaken = len(ahead_t) > 0
        rows.append(dict(seed=s, behavior=env.behavior, gap0=env.gap0,
                         steps=len(t_arr), reason=env.term_reason,
                         overtaken=overtaken,
                         t_overtake=ahead_t[0] if overtaken else np.nan,
                         mean_v=np.mean(lg["v_f"]), log=lg))
        logs.append(lg)
    return rows, logs


def eval_cmd(args):
    seeds = args.seeds if args.seeds else list(range(2000, 2010))
    print(f"evaluating {len(seeds)} fixed seeds (domain_randomize={args.domain_randomize})")

    policies = {"rule-based": lambda o: baseline_action_ot(o)}
    if not args.rule_only:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        policies["RL(PPO)"] = lambda o: model.predict(o, deterministic=True)[0]
    all_logs = {}
    for name, pol in policies.items():
        rows, logs = rollout(pol, seeds, args.domain_randomize)
        all_logs[name] = logs
        succ = sum(r["overtaken"] for r in rows)
        col = sum(r["reason"] == "collision" for r in rows)
        off = sum(r["reason"] == "offtrack" for r in rows)
        lost = sum(r["reason"] == "lost" for r in rows)
        t_ot = [r["t_overtake"] for r in rows if r["overtaken"]]
        print(f"{name:12s} overtake={succ}/{len(rows)}  collision={col}  "
              f"offtrack={off}  lost={lost}  "
              f"t_overtake={np.mean(t_ot) if t_ot else float('nan'):5.1f}s  "
              f"mean_v={np.mean([r['mean_v'] for r in rows]):.2f} m/s")
        if args.verbose:
            for r in rows:
                print(f"    seed={r['seed']} {r['behavior']:9s} gap0={r['gap0']:.2f} "
                      f"steps={r['steps']:4d} {r['reason']:8s} "
                      f"overtaken={r['overtaken']} t_ot={r['t_overtake']:.1f}s")
                if r["reason"] not in ("timeout", "success"):
                    lg = r["log"]
                    print("      last steps before termination (t, v_f, lane, e_lat[cm]):")
                    for k in range(max(0, len(lg["t"]) - 30), len(lg["t"]), 3):
                        print(f"        t={lg['t'][k]:6.2f} v={lg['v_f'][k]:.2f} "
                              f"lane={lg['lane'][k]} e_lat={lg['e_lat'][k]*100:+6.1f}")
    if "RL(PPO)" in all_logs:
        plot_episode(all_logs["RL(PPO)"][0], all_logs["rule-based"][0], args.out)
    else:
        plot_episode(all_logs["rule-based"][0], None, args.out)


def plot_episode(lg_rl, lg_rb, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(lg_rl["t"], lg_rl["delta"], label="policy")
    if lg_rb is not None:
        ax1.plot(lg_rb["t"], lg_rb["delta"], label="rule-based")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("delta [m]  (+ = ahead)")
    ax1.set_title("progress difference (same seed)"); ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(lg_rl["t"], lg_rl["v_f"], label="policy v")
    ax2.plot(lg_rl["t"], lg_rl["v_l"], "k--", lw=1, label="leader v")
    ax2.plot(lg_rl["t"], lg_rl["lane"], label="policy lane (1=inner)")
    if lg_rb is not None:
        ax2.plot(lg_rb["t"], lg_rb["lane"], "--", label="rule lane")
    ax2.set_xlabel("t [s]"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"figure saved -> {out}")


def pretrain(args):
    """Behavior-cloning warm start (LfD): imitate the rule-based baseline so
    PPO fine-tuning starts from a passing-capable policy instead of the
    'follow forever' local optimum (the exploration valley is too deep for
    reward shaping alone)."""
    import torch
    from stable_baselines3 import PPO

    print(f"collecting {args.n_demos} rule-based demonstrations ...")
    obs_buf, act_buf = [], []
    for i in range(args.n_demos):
        env = OvertakeEnv(domain_randomize=True)
        obs, _ = env.reset(seed=30_000 + i)
        while True:
            a = baseline_action_ot(obs)
            obs_buf.append(obs.copy())
            act_buf.append(a.copy())
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    dev = "cpu"   # BC is a tiny supervised job; keep everything on CPU
    obs_t = torch.as_tensor(np.array(obs_buf), dtype=torch.float32).to(dev)
    act_t = torch.as_tensor(np.array(act_buf), dtype=torch.float32).to(dev)
    print(f"demo dataset: {len(obs_t)} transitions")

    model = PPO("MlpPolicy", OvertakeEnv(domain_randomize=True),
                policy_kwargs=dict(net_arch=[128, 128]),
                seed=args.seed, verbose=0, device=dev)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n = len(obs_t)
    for ep in range(args.bc_epochs):
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for b in range(0, n, 8192):
            idx = perm[b:b + 8192]
            dist = model.policy.get_distribution(obs_t[idx])
            loss = -dist.log_prob(act_t[idx]).mean()   # max log-likelihood of demos
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        print(f"  BC epoch {ep+1}/{args.bc_epochs}  loss={tot/cnt:.4f}")
    model.save("ckpt_ot/bc_model")
    print("BC model -> ckpt_ot/bc_model.zip")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pretrain")
    pp.add_argument("--n-demos", type=int, default=150)
    pp.add_argument("--bc-epochs", type=int, default=10)
    pp.add_argument("--seed", type=int, default=0)
    pp.set_defaults(func=pretrain)
    pt = sub.add_parser("train")
    pt.add_argument("--timesteps", type=int, default=5_000_000)
    pt.add_argument("--n-envs", type=int, default=32)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--load", default=None)
    pt.add_argument("--easy", action="store_true",
                    help="curriculum stage 1: constant slow leader, short gap")
    pt.set_defaults(func=train)
    pe = sub.add_parser("eval")
    pe.add_argument("--model", default="ckpt_ot/best_model.zip")
    pe.add_argument("--seeds", type=int, nargs="*", default=None)
    pe.add_argument("--domain-randomize", action="store_true")
    pe.add_argument("--out", default="eval_overtake.png")
    pe.add_argument("-v", "--verbose", action="store_true")
    pe.add_argument("--rule-only", action="store_true",
                    help="health check: evaluate the rule baseline only, no model needed")
    pe.set_defaults(func=eval_cmd)
    args = ap.parse_args()
    args.func(args)
