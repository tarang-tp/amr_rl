"""
scripts/smoke_test.py

End-to-end pipeline smoke test. Verifies every major component wires together
correctly without requiring a full training run.

Runs in ~30 seconds on CPU. Use this before committing or after merging.

Exit code 0 = all checks passed.
Exit code 1 = something broke (error printed above).

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --verbose
"""

from __future__ import annotations

import sys
import time
import traceback
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def check(name: str, fn, verbose: bool = False):
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        msg = f" ({result})" if isinstance(result, str) else ""
        print(f"  {PASS} {name}{msg}  [{elapsed:.2f}s]")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {FAIL} {name}  [{elapsed:.2f}s]")
        if verbose:
            traceback.print_exc()
        else:
            print(f"       {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    v = args.verbose

    failures = 0
    total = 0

    def run(section: str, checks: list[tuple]):
        nonlocal failures, total
        print(f"\n{section}")
        for name, fn in checks:
            total += 1
            if not check(name, fn, verbose=v):
                failures += 1

    # ── 1. PK/PD ─────────────────────────────────────────────────────────────
    from simulator.pkpd.pharmacokinetics import PKParams, PDParams, PKPDModel

    def _pkpd_one_step():
        m = PKPDModel(pk=PKParams(), pd=PDParams(ec50=0.1), rng=np.random.default_rng(0))
        m.reset()
        load, c, info = m.step_day(dose=400.0, bacterial_load=1e7, resistance_level=0.0)
        assert load < 1e7, f"load={load:.2e} should decrease"
        assert c > 0
        return f"load {1e7:.0e} -> {load:.2e}"

    def _pkpd_zero_dose_grows():
        m = PKPDModel(pk=PKParams(), pd=PDParams(ec50=0.1), rng=np.random.default_rng(0))
        m.reset()
        load, _, _ = m.step_day(dose=0.0, bacterial_load=1e6, resistance_level=0.0)
        assert load > 1e6
        return "ok"

    run("1. PK/PD model", [
        ("single step reduces load (dose=400)", _pkpd_one_step),
        ("zero dose allows growth", _pkpd_zero_dose_grows),
    ])

    # ── 2. Reward function ────────────────────────────────────────────────────
    from simulator.reward.reward_fn import RewardFunction, RewardConfig

    def _reward_clearance():
        rf = RewardFunction()
        r, info = rf(bacterial_load=500, prev_load=1e6, dose=1.0,
                     resistance_level=0.0, prev_resistance_level=0.0,
                     in_msw=False, done=True)
        assert info["clearance"] == 5.0
        return f"r={r:.3f}"

    def _reward_msw():
        rf = RewardFunction()
        _, i1 = rf(bacterial_load=1e5, prev_load=1e6, dose=1.0,
                   resistance_level=0.0, prev_resistance_level=0.0,
                   in_msw=True, done=False)
        _, i2 = rf(bacterial_load=1e5, prev_load=1e6, dose=1.0,
                   resistance_level=0.0, prev_resistance_level=0.0,
                   in_msw=False, done=False)
        assert i1["msw"] < i2["msw"]
        return "msw penalty active"

    run("2. Reward function", [
        ("clearance bonus fires on done+cleared", _reward_clearance),
        ("MSW penalty applied when in window", _reward_msw),
    ])

    # ── 3. Environment ────────────────────────────────────────────────────────
    from simulator.envs.amr_env import AMREnv, DOSE_LEVELS

    def _env_reset_step():
        env = AMREnv(seed=0)
        obs, _ = env.reset()
        assert obs.shape == (11,)
        obs2, r, term, trunc, info = env.step(4)
        assert "bacterial_load" in info
        env.close()
        return f"obs={obs.shape}, r={r:.3f}"

    def _env_clears_with_max_dose():
        env = AMREnv(seed=0)
        env.reset()
        cleared = False
        for _ in range(14):
            _, _, term, trunc, info = env.step(4)
            if info["cleared"]:
                cleared = True
            if term or trunc:
                break
        env.close()
        assert cleared, "Max dose should clear within 14 days"
        return "cleared"

    def _env_1000_random_steps():
        env = AMREnv(seed=1)
        rng = np.random.default_rng(1)
        env.reset()
        steps = 0
        episodes = 0
        while steps < 1000:
            action = int(rng.integers(0, 5))
            _, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                env.reset()
                episodes += 1
        env.close()
        return f"{steps} steps, {episodes} episodes"

    run("3. AMR environment", [
        ("reset + step returns correct shapes", _env_reset_step),
        ("max dose clears infection", _env_clears_with_max_dose),
        ("1000 random steps without crash", _env_1000_random_steps),
    ])

    # ── 4. Resistance model ───────────────────────────────────────────────────
    from resistance.model.resistance_model import AdversarialResistanceModel

    def _resistance_model_call():
        m = AdversarialResistanceModel(obs_dim=14, device="cpu")
        rng = np.random.default_rng(0)
        obs = np.zeros(11, dtype=np.float32)
        results = [m(obs, 0.0, 200.0, False, rng) for _ in range(20)]
        assert all(0.0 <= r <= 4.0 for r in results)
        return f"transitions: {set(results)}"

    def _resistance_adversarial_update():
        m = AdversarialResistanceModel(obs_dim=14, device="cpu")
        rng = np.random.default_rng(0)
        obs = np.zeros(11, dtype=np.float32)
        for _ in range(10):
            m(obs, 0.0, 200.0, False, rng)
        info = m.adversarial_update(policy_losses=[-2.0])
        assert "adversarial_loss" in info
        assert len(m._episode_log) == 0
        return f"loss={info['adversarial_loss']:.4f}"

    def _resistance_pretrain():
        from resistance.data_loaders.patric_loader import MockPATRICLoader
        loader = MockPATRICLoader(seed=0)
        train_loader, _ = loader.load(n=128)
        # Pretraining uses gene features only (no obs prefix), so obs_dim = N_GENES
        from resistance.data_loaders.patric_loader import N_GENES
        m = AdversarialResistanceModel(obs_dim=N_GENES, device="cpu")
        losses = []
        for X, y in train_loader:
            losses.append(m.pretrain_step(X, y))
        assert len(losses) > 0
        return f"{len(losses)} batches, final_loss={losses[-1]:.4f}"

    run("4. Resistance model", [
        ("__call__ returns valid resistance levels", _resistance_model_call),
        ("adversarial_update runs and clears log", _resistance_adversarial_update),
        ("pretrain_step on mock PATRIC data", _resistance_pretrain),
    ])

    # ── 5. Baselines ──────────────────────────────────────────────────────────
    from baselines.baselines import CyclingHeuristic, ContextualBanditPolicy, MaxDosePolicy

    def _baselines_predict():
        obs = np.zeros(11, dtype=np.float32)
        c = CyclingHeuristic()
        b = ContextualBanditPolicy()
        m = MaxDosePolicy()
        actions = [c.predict(obs), b.predict(obs), m.predict(obs)]
        assert all(0 <= a <= 4 for a in actions)
        return f"cycling={actions[0]}, bandit={actions[1]}, max={actions[2]}"

    run("5. Baselines", [
        ("all baselines return valid actions", _baselines_predict),
    ])

    # ── 6. Adversarial env injection ──────────────────────────────────────────
    def _adversarial_injection():
        """Verify resistance model plugs into env and modulates transitions."""
        call_count = [0]

        def counting_model(obs, resistance_level, dose, in_msw, rng):
            call_count[0] += 1
            return resistance_level  # static — just count calls

        env = AMREnv(resistance_model=counting_model, seed=0)
        env.reset()
        for _ in range(5):
            _, _, term, trunc, _ = env.step(2)
            if term or trunc:
                break
        env.close()
        assert call_count[0] >= 5
        return f"model called {call_count[0]}x"

    run("6. Adversarial injection", [
        ("custom resistance model called each step", _adversarial_injection),
    ])

    # ── 7. Co-trainer (short smoke) ───────────────────────────────────────────
    def _co_trainer_init():
        from training.adversarial.co_trainer import AdversarialCoTrainer
        trainer = AdversarialCoTrainer(
            env_kwargs={"max_episode_steps": 7},
            policy_kwargs={"n_steps": 64, "batch_size": 32, "n_epochs": 2,
                           "learning_rate": 3e-4},
            adversary_kwargs={"obs_dim": 14, "hidden_dims": [64, 64]},
            total_timesteps=256,
            log_dir="/tmp/amr_smoke_log/",
            checkpoint_dir="/tmp/amr_smoke_ckpt/",
            eval_freq=256,
            n_eval_episodes=2,
            seed=0,
        )
        history = trainer.train()
        assert "policy_reward" in history
        return f"{len(history['policy_reward'])} iters"

    run("7. Co-trainer (256 steps)", [
        ("co-trainer initialises and runs", _co_trainer_init),
    ])

    # ── 8. Evaluation metrics ────────────────────────────────────────────────
    def _eval_metrics():
        from evaluation.metrics.eval_metrics import evaluate_policy, generate_ood_profiles
        from baselines.baselines import CyclingHeuristic

        env = AMREnv(seed=0)
        policy = CyclingHeuristic()
        results = evaluate_policy(policy, env, n_episodes=10, policy_name="cycling")
        assert results.n_episodes == 10
        assert 0.0 <= results.clearance_rate <= 1.0
        env.close()

        profiles = generate_ood_profiles(5)
        assert len(profiles) == 5
        return f"clearance={results.clearance_rate:.0%}, mean_r={results.mean_reward:.2f}"

    run("8. Evaluation metrics", [
        ("evaluate_policy runs 10 episodes", _eval_metrics),
    ])

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = total - failures
    color = "\033[92m" if failures == 0 else "\033[91m"
    print(f"\n{color}{'-'*50}\033[0m")
    print(f"{color}  {passed}/{total} checks passed\033[0m")
    if failures:
        print(f"\033[91m  {failures} failures — run with --verbose for tracebacks\033[0m")
    print()

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
