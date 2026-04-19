"""
baselines/baselines.py

Baseline treatment policies for comparison against the adversarially-trained PPO agent.

Baselines (per paper scope):
  1. FixedResistancePolicy  — PPO trained against a *static* (non-updating) resistance model
  2. CyclingHeuristic       — deterministic antibiotic cycling every k days
  3. ContextualBanditPolicy — epsilon-greedy bandit, no resistance modelling
  4. MaxDosePolicy          — always doses at maximum (clinical worst-case)

All expose a predict(obs) -> action interface compatible with AMREnv.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BasePolicy(ABC):
    """Minimal interface for all policies."""

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        ...

    def reset(self) -> None:
        pass


# ── 1. Fixed resistance PPO (trained separately, loaded at eval) ─────────────

class FixedResistancePPOWrapper(BasePolicy):
    """
    Wraps an SB3 PPO model trained against a *fixed* (non-adaptive) resistance model.
    Use this to demonstrate the OOD generalisation gap vs adversarial policy.
    """

    def __init__(self, sb3_model):
        self.model = sb3_model

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)


# ── 2. Antibiotic cycling heuristic ─────────────────────────────────────────

class CyclingHeuristic(BasePolicy):
    """
    Cycle through dose levels on a fixed schedule.
    Clinically common — switches antibiotic (here modelled as dose level)
    every `period` days to avoid sustained selection pressure.
    """

    DOSE_SCHEDULE = [1.0, 1.5, 1.0, 0.5]  # mg/kg per cycle position

    def __init__(self, period: int = 3):
        self.period = period
        self._day = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        # Map to dose level index (closest match in DOSE_LEVELS = [0,0.5,1,1.5,2])
        dose = self.DOSE_SCHEDULE[(self._day // self.period) % len(self.DOSE_SCHEDULE)]
        self._day += 1
        # Convert to discrete action index
        dose_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
        return int(np.argmin(np.abs(np.array(dose_levels) - dose)))

    def reset(self) -> None:
        self._day = 0


# ── 3. Contextual bandit ─────────────────────────────────────────────────────

class ContextualBanditPolicy(BasePolicy):
    """
    Epsilon-greedy contextual bandit with a linear value function per action.
    Does not model resistance — purely reactive to immediate bacterial load.

    Q(s, a) = w_a · s  (linear approximation)
    """

    def __init__(
        self,
        obs_dim: int = 11,
        n_actions: int = 5,
        epsilon: float = 0.1,
        lr: float = 0.01,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((n_actions, obs_dim))  # Q(s,a) = w_a · s

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q_vals = self.weights @ obs.flatten()
        return int(np.argmax(q_vals))

    def update(self, obs: np.ndarray, action: int, reward: float) -> None:
        """Online TD(0) update."""
        q_pred = self.weights[action] @ obs.flatten()
        error = reward - q_pred
        self.weights[action] += self.lr * error * obs.flatten()


# ── 4. Max-dose policy ───────────────────────────────────────────────────────

class MaxDosePolicy(BasePolicy):
    """Always administers the maximum dose. Worst-case for toxicity/resistance."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return 4  # Index of 2.0 mg/kg in DOSE_LEVELS


# ── 5. Zero-dose (no treatment) — failure mode reference ────────────────────

class NoDosePolicy(BasePolicy):
    """Never administers any drug. Reference for minimum possible performance."""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return 0  # Index of 0.0 mg/kg


# ── Registry ─────────────────────────────────────────────────────────────────

BASELINE_REGISTRY: dict[str, type[BasePolicy]] = {
    "cycling": CyclingHeuristic,
    "bandit": ContextualBanditPolicy,
    "max_dose": MaxDosePolicy,
    "no_dose": NoDosePolicy,
}


def make_baseline(name: str, **kwargs) -> BasePolicy:
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](**kwargs)
