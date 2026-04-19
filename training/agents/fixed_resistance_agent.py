"""
training/agents/fixed_resistance_agent.py

Trains a PPO agent against a *fixed* (non-updating) resistance model.
This is the key baseline for the paper's OOD generalisation claim:

  Claim: adversarially co-trained policy generalises better to unseen
  resistance profiles than a policy trained against a static adversary.

Usage:
  python -m training.agents.fixed_resistance_agent \
      --total_timesteps 500000 \
      --resistance_mode static      # static | random | worst_case
      --output checkpoints/fixed_ppo.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator.envs.amr_env import AMREnv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ── Fixed resistance model variants ──────────────────────────────────────────

def make_static_resistance(p_increase: float = 0.02):
    """Always transitions with the same fixed probability — no adaptation."""
    rng = np.random.default_rng(0)

    def _model(obs, resistance_level, dose, in_msw, episode_rng):
        if resistance_level < 4.0 and rng.random() < p_increase:
            return min(resistance_level + 1.0, 4.0)
        return resistance_level

    return _model


def make_random_resistance(seed: int = 42):
    """
    Random resistance model — samples a fresh fixed probability each episode.
    Exposes the policy to more varied resistance trajectories than static,
    but still non-adaptive (no response to dosing strategy).
    """
    rng = np.random.default_rng(seed)
    state = {"p": 0.02}

    def _reset_episode():
        state["p"] = float(rng.uniform(0.01, 0.15))

    def _model(obs, resistance_level, dose, in_msw, episode_rng):
        if resistance_level < 4.0 and episode_rng.random() < state["p"]:
            return min(resistance_level + 1.0, 4.0)
        return resistance_level

    _model.reset_episode = _reset_episode
    return _model


def make_worst_case_resistance():
    """
    Worst-case fixed model — always transitions if dose is in MSW.
    Maximally punishing but not adaptive to the specific policy.
    """
    def _model(obs, resistance_level, dose, in_msw, episode_rng):
        if resistance_level < 4.0 and in_msw:
            return min(resistance_level + 1.0, 4.0)
        return resistance_level

    return _model


RESISTANCE_MODES = {
    "static":     make_static_resistance,
    "random":     make_random_resistance,
    "worst_case": make_worst_case_resistance,
}


# ── Training ──────────────────────────────────────────────────────────────────

def train_fixed_resistance_ppo(
    resistance_mode: str = "static",
    total_timesteps: int = 1_000_000,
    env_kwargs: dict | None = None,
    policy_kwargs: dict | None = None,
    log_dir: str = "runs/fixed_ppo/",
    checkpoint_dir: str = "checkpoints/",
    eval_freq: int = 50_000,
    seed: int = 42,
    device: str = "cpu",
) -> PPO:
    env_kwargs = env_kwargs or {}
    policy_kwargs = policy_kwargs or {}

    resistance_fn = RESISTANCE_MODES[resistance_mode]()

    def _make_env():
        return AMREnv(resistance_model=resistance_fn, seed=seed, **env_kwargs)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_env = VecMonitor(DummyVecEnv([_make_env]), str(log_path))
    eval_env  = VecMonitor(DummyVecEnv([_make_env]), str(log_path / "eval"))

    ppo_kw = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
    }
    ppo_kw.update(policy_kwargs)

    model = PPO("MlpPolicy", train_env, verbose=1, seed=seed, device=device, **ppo_kw)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_path / f"fixed_{resistance_mode}_best"),
        log_path=str(log_path),
        eval_freq=eval_freq,
        n_eval_episodes=50,
        deterministic=True,
        verbose=0,
    )

    log.info(f"Training fixed-resistance PPO (mode={resistance_mode}, "
             f"steps={total_timesteps:,}, device={device})")
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    out_path = str(ckpt_path / f"fixed_ppo_{resistance_mode}.zip")
    model.save(out_path)
    log.info(f"Saved to {out_path}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resistance_mode", type=str, default="static",
                        choices=list(RESISTANCE_MODES))
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        env_kwargs = {
            "drug": cfg["env"]["drug"],
            "max_episode_steps": cfg["env"]["max_episode_steps"],
            "bacterial_load_init": cfg["env"]["bacterial_load_init"],
            "target_load": cfg["env"]["target_load"],
        }
        ts = args.total_timesteps or cfg["policy"]["total_timesteps"]
    except FileNotFoundError:
        env_kwargs = {}
        ts = args.total_timesteps or 1_000_000

    train_fixed_resistance_ppo(
        resistance_mode=args.resistance_mode,
        total_timesteps=ts,
        env_kwargs=env_kwargs,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
