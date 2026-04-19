"""
training/adversarial/co_trainer.py

Adversarial co-training: PPO treatment policy vs adaptive resistance model.

Training loop structure:
  for each outer iteration:
    1. Update resistance model (adversary) to maximise resistance emergence
    2. Run N PPO update steps (policy rollouts with current adversary)
    3. Evaluate policy vs adversary and static baselines
    4. Log and checkpoint

The adversary acts as the pathogen's mutation "strategy". Its goal is to
find resistance transitions that defeat the current dosing policy. The
policy must generalise to resist this.

Key paper claim:
  Policies trained against an adaptive adversary generalise better to
  unseen resistance profiles than policies trained against fixed/static
  resistance models.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from simulator.envs.amr_env import AMREnv
from simulator.reward.reward_fn import RewardConfig
from resistance.model.resistance_model import AdversarialResistanceModel

log = logging.getLogger(__name__)


class AdversarialCoTrainer:
    """
    Orchestrates minimax co-training between:
      - `policy`:   SB3 PPO agent (treatment policy)
      - `adversary`: AdversarialResistanceModel (pathogen evolution strategy)
    """

    def __init__(
        self,
        env_kwargs: dict,
        policy_kwargs: dict,
        adversary_kwargs: dict,
        total_timesteps: int = 2_000_000,
        co_train_ratio: int = 4,        # policy steps per adversary update
        adversary_update_freq: int = 4,
        log_dir: str = "runs/",
        checkpoint_dir: str = "checkpoints/",
        eval_freq: int = 50_000,
        n_eval_episodes: int = 50,
        seed: int = 42,
        device: str = "cpu",
        use_wandb: bool = False,
        ec50_predictor_path: Optional[str] = None,
    ):
        self.env_kwargs = env_kwargs
        self.policy_kwargs = policy_kwargs
        self.adversary_kwargs = adversary_kwargs
        self.total_timesteps = total_timesteps
        self.co_train_ratio = co_train_ratio
        self.adversary_update_freq = adversary_update_freq
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.device = device
        self.use_wandb = use_wandb
        self.ec50_predictor_path = ec50_predictor_path

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optionally load EC50 predictor from pretrained checkpoint
        self._ec50_predictor = None
        if ec50_predictor_path:
            from resistance.model.resistance_model import EC50Predictor
            self._ec50_predictor = EC50Predictor.load_from_checkpoint(
                ec50_predictor_path, device=device
            )
            log.info(f"Loaded EC50 predictor from {ec50_predictor_path}")

        # Build adversary
        self.adversary = AdversarialResistanceModel(
            **adversary_kwargs, device=device
        )

        # Build envs
        self.train_env = self._make_env(adversary=self.adversary)
        self.eval_env = self._make_env(adversary=None)   # static for fair eval

        # Build policy
        self.policy = PPO(
            "MlpPolicy",
            self.train_env,
            verbose=0,
            seed=seed,
            device=device,
            **policy_kwargs,
        )

        self._step = 0
        self._episode_policy_losses: list[float] = []
        self._adversary_episode_logs: list[list[dict]] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def train(self) -> dict:
        """Run full adversarial co-training loop."""
        log.info("Starting adversarial co-training.")
        log.info(f"  Total timesteps: {self.total_timesteps:,}")
        log.info(f"  Co-train ratio:  {self.co_train_ratio}")
        log.info(f"  Device:          {self.device}")

        callbacks = self._build_callbacks()
        history: dict = {"policy_reward": [], "adversary_loss": [], "timesteps": [],
                         "ec50_multiplier": []}

        chunk = max(self.policy_kwargs.get("n_steps", 2048) * self.co_train_ratio, 2048)
        steps_done = 0

        while steps_done < self.total_timesteps:
            t0 = time.time()

            # ── 1. Swap in the current adversary ───────────────────────────
            self._inject_adversary(self.adversary)

            # ── 2. Policy rollout + PPO update ─────────────────────────────
            self.policy.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                callback=callbacks,
                log_interval=None,
            )
            steps_done += chunk

            # ── 3. Adversarial update ──────────────────────────────────────
            if self._episode_policy_losses:
                adv_info = self.adversary.adversarial_update(
                    policy_losses=self._episode_policy_losses,
                    episode_logs=self._adversary_episode_logs,
                )
                self._episode_policy_losses = []
                self._adversary_episode_logs = []
            else:
                adv_info = {"adversarial_loss": 0.0}

            # ── 4. Eval & log ──────────────────────────────────────────────
            mean_rew = self._quick_eval()
            elapsed = time.time() - t0

            log.info(
                f"  Steps {steps_done:>8,}/{self.total_timesteps:,} | "
                f"MeanRew={mean_rew:+.3f} | "
                f"AdvLoss={adv_info['adversarial_loss']:.4f} | "
                f"Time={elapsed:.1f}s"
            )

            # Log mean EC50 multiplier across envs this iteration
            try:
                mults = self.train_env.get_attr("_current_ec50_multiplier")
                mean_ec50 = float(np.mean(mults))
            except Exception:
                mean_ec50 = 1.0

            history["policy_reward"].append(mean_rew)
            history["adversary_loss"].append(adv_info["adversarial_loss"])
            history["timesteps"].append(steps_done)
            history["ec50_multiplier"].append(mean_ec50)

            log.info(f"    EC50 multiplier={mean_ec50:.3f}")

            if self.use_wandb:
                self._log_wandb(steps_done, mean_rew, adv_info)

            # ── 5. Checkpoint ─────────────────────────────────────────────
            if steps_done % (self.eval_freq * 2) < chunk:
                self._save_checkpoint(steps_done)

        log.info("Co-training complete.")
        self._save_checkpoint(steps_done, final=True)
        return history

    def load_pretrained_adversary(self, path: str) -> None:
        """Load pretrained (PATRIC/CARD) resistance model weights."""
        self.adversary.load(path)
        log.info(f"Loaded pretrained adversary from {path}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_env(self, adversary=None) -> VecMonitor:
        ec50_predictor = self._ec50_predictor  # capture for closure

        def _env_fn():
            env = AMREnv(
                resistance_model=adversary,
                ec50_predictor=ec50_predictor,
                **self.env_kwargs,
            )
            return env

        vec = DummyVecEnv([_env_fn])
        return VecMonitor(vec, str(self.log_dir))

    def _inject_adversary(self, adversary) -> None:
        """Update the resistance model inside the training env."""
        for env in self.train_env.envs:
            unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
            unwrapped.resistance_model = adversary

    def _quick_eval(self, n_episodes: int = 20) -> float:
        """Fast in-loop evaluation on the static eval env."""
        rewards = []
        obs = self.eval_env.reset()
        ep_reward = 0.0
        ep_count = 0

        while ep_count < n_episodes:
            action, _ = self.policy.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            ep_reward += float(reward[0])
            if done[0]:
                rewards.append(ep_reward)
                ep_reward = 0.0
                ep_count += 1
                obs = self.eval_env.reset()

        return float(np.mean(rewards)) if rewards else 0.0

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        tag = "final" if final else f"step_{step:08d}"
        policy_path = self.checkpoint_dir / f"policy_{tag}.zip"
        adv_path = self.checkpoint_dir / f"adversary_{tag}.pt"
        self.policy.save(str(policy_path))
        self.adversary.save(str(adv_path))
        log.info(f"Saved checkpoint: {tag}")

    def _build_callbacks(self) -> list[BaseCallback]:
        eval_cb = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.checkpoint_dir / "best"),
            log_path=str(self.log_dir),
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            verbose=0,
        )
        policy_loss_cb = _PolicyLossCapture(self)
        return [eval_cb, policy_loss_cb]

    def _log_wandb(self, step: int, mean_rew: float, adv_info: dict) -> None:
        try:
            import wandb
            wandb.log({
                "policy/mean_reward": mean_rew,
                "adversary/loss": adv_info["adversarial_loss"],
                "train/timestep": step,
            })
        except Exception:
            pass


class _PolicyLossCapture(BaseCallback):
    """Captures per-episode policy losses for the adversarial update."""

    def __init__(self, trainer: AdversarialCoTrainer):
        super().__init__(verbose=0)
        self.trainer = trainer

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("episode"):
                ep_rew = info["episode"]["r"]
                self.trainer._episode_policy_losses.append(-ep_rew)
                # Robustly unwrap through VecMonitor -> DummyVecEnv -> gym.Env
                try:
                    envs_list = self.training_env.envs
                except AttributeError:
                    venv = getattr(self.training_env, "venv", None)
                    envs_list = getattr(venv, "envs", []) if venv is not None else []
                if envs_list:
                    inner = envs_list[0]
                    unwrapped = inner.unwrapped if hasattr(inner, "unwrapped") else inner
                    if hasattr(unwrapped, "resistance_model"):
                        rm = unwrapped.resistance_model
                        if hasattr(rm, "_episode_log"):
                            self.trainer._adversary_episode_logs.append(rm._episode_log.copy())
                            rm._episode_log = []
        return True
