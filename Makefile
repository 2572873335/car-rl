# car-rl — convenience targets
#
# `make demo` runs the FULL pipeline end-to-end in a few minutes with SMALL
# step counts, so you can see it work.  It is a functional smoke test, NOT a
# result reproduction: the short runs intentionally violate the project's
# ">= 300 gradient updates" rule.  For real results use `make reproduce`.

PY ?= uv run python

.PHONY: help check demo demo-follow demo-overtake reproduce clean

help:
	@echo "make check          - health checks only (run these first)"
	@echo "make demo           - full pipeline, small step counts, ~5 min (smoke test)"
	@echo "make demo-follow    - following task only, small run"
	@echo "make demo-overtake  - overtaking task only, small run"
	@echo "make reproduce      - full-length training (hours; see report appendix A)"

## health checks: the project's iron rule 3 — always run before training
check:
	$(PY) car_following_sim.py
	$(PY) train_ot.py eval --rule-only --out /tmp/car_rl_ruleonly.png

## full smoke test: check -> short follow train+eval -> short overtake train+eval
demo: check demo-follow demo-overtake
	@echo "demo complete — figures under ckpt/ and /tmp; see README for real runs"

demo-follow:
	$(PY) train_ppo.py train --easy --timesteps 40000 --n-envs 8 --seed 0
	$(PY) train_ppo.py eval --model ckpt/final_model.zip \
		--out /tmp/car_rl_demo_follow.png

demo-overtake:
	$(PY) train_ot.py pretrain --n-demos 40 --bc-epochs 2 --seed 0
	$(PY) train_ot.py eval --model ckpt_ot/bc_model.zip \
		--out /tmp/car_rl_demo_overtake_bc.png
	$(PY) train_ot.py train --timesteps 40000 --n-envs 8 --seed 0 \
		--load ckpt_ot/bc_model.zip
	$(PY) train_ot.py eval --model ckpt_ot/final_model.zip \
		--out /tmp/car_rl_demo_overtake.png

## full-length reproduction (matches the report).  Hours on CPU.
reproduce: check
	$(PY) train_ppo.py train --easy --timesteps 2500000 --n-envs 32 --seed 0
	$(PY) train_ppo.py train --timesteps 5000000 --n-envs 32 --seed 0 \
		--load ckpt/best_model.zip
	$(PY) train_ppo.py eval --model ckpt/final_model.zip --domain-randomize -v \
		--out results/reproduce_follow.png
	$(PY) train_ot.py pretrain --n-demos 300 --bc-epochs 10
	$(PY) train_ot.py train --timesteps 5000000 --n-envs 32 --seed 0 \
		--load ckpt_ot/bc_model.zip
	$(PY) train_ot.py eval --model ckpt_ot/final_model.zip -v \
		--out results/reproduce_overtake.png

clean:
	rm -f /tmp/car_rl_*.png
