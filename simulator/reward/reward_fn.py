"""
simulator/reward/reward_fn.py

Reward shaping for the AMR treatment environment.

Design goals (per TMLR paper scope):
  1. Incentivise bacterial clearance
  2. Penalise toxicity (dose magnitude)
  3. Penalise dosing in the mutant selection window (MSW) — biologically grounded
  4. Penalise resistance emergence events
  5. Small per-step penalty on log(bacterial_load) to encourage speed
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    w_clearance: float = 5.0       # terminal bonus for reaching target_load
    w_load: float = -0.01          # per-step, multiplied by log10(N/N0)
    w_dose: float = -0.005         # per-unit-dose toxicity
    w_resistance: float = -2.0     # per resistance-level increase event
    msw_penalty: float = -1.5      # scaling factor for MSW exposure
    target_load: float = 1e3       # CFU/mL — clinical clearance threshold
    initial_load: float = 1e8      # CFU/mL — episode starting load


class RewardFunction:
    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()

    def __call__(
        self,
        *,
        bacterial_load: float,
        prev_load: float,
        dose: float,
        resistance_level: float,
        prev_resistance_level: float,
        in_msw: bool,
        done: bool,
    ) -> tuple[float, dict]:
        """
        Compute reward for a single environment step.

        Returns (reward, info_dict).
        """
        cfg = self.cfg
        reward = 0.0
        components = {}

        # 1. Bacterial load reduction signal (log-scale, bounded)
        load_signal = cfg.w_load * np.log10(
            max(bacterial_load, 1.0) / cfg.initial_load
        )
        reward += load_signal
        components["load"] = load_signal

        # 2. Dose toxicity
        dose_penalty = cfg.w_dose * dose
        reward += dose_penalty
        components["dose"] = dose_penalty

        # 3. Resistance emergence
        resistance_delta = resistance_level - prev_resistance_level
        resist_penalty = cfg.w_resistance * max(resistance_delta, 0.0)
        reward += resist_penalty
        components["resistance"] = resist_penalty

        # 4. Mutant selection window penalty (proportional to MSW exposure time)
        msw_pen = cfg.msw_penalty if in_msw else 0.0
        reward += msw_pen
        components["msw"] = msw_pen

        # 5. Terminal clearance bonus
        clearance_bonus = 0.0
        if done and bacterial_load <= cfg.target_load:
            clearance_bonus = cfg.w_clearance
            reward += clearance_bonus
        components["clearance"] = clearance_bonus

        components["total"] = reward
        return float(reward), components
