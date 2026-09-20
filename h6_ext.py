import warnings
warnings.filterwarnings('ignore')
import numpy as np
from stable_baselines3 import PPO
from overtake_env import OvertakeEnv

seeds = list(range(2000, 2030))   # 30 seeds
print('=== H6 boundary check: manual-BC extended to 30 seeds ===')
res = {}
for n in [50, 100, 200, 300]:
    m = PPO.load('ckpt_offline/manualbc_d%d.zip' % n)
    succ = col = 0
    t_ots = []
    for s in seeds:
        env = OvertakeEnv(domain_randomize=False)
        obs, _ = env.reset(seed=s)
        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        lg = env.log
        t = np.array(lg['t']); d = np.array(lg['delta'])
        ah = t[d > 0.05]
        if len(ah) > 0:
            succ += 1
            t_ots.append(ah[0])
        if env.term_reason == 'collision':
            col += 1
    res[n] = succ
    print('H6EXT d=%3d: success=%d/30 collision=%d t_ot=%.1fs' % (
        n, succ, col, np.mean(t_ots) if t_ots else float('nan')))

vals = list(res.values())
print('---')
print('H6 verdict (30-seed): max=%d/30 min=%d/30 spread=%d/30 (threshold <=2/10 => 6/30)' % (
    max(vals), min(vals), max(vals) - min(vals)))
print('H6 holds at 30 seeds:', (max(vals) - min(vals)) <= 6)
