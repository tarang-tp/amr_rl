"""
evaluation/metrics/eval_metrics.py

Evaluation suite for the TMLR paper.

Key metrics:
  1. OOD generalisation gap
     — performance on held-out resistance profiles not seen during training
  2. Time-to-resistance hazard function (survival analysis via lifelines)
  3. Policy convergence diagnostics
  4. Treatment success rate and bacterial clearance statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    """Records from a single evaluation episode."""
    policy_name: str
    resistance_profile_id: str
    total_reward: float
    cleared: bool
    clearance_day: Optional[int]
    final_resistance: float
    final_load: float
    day_resistance_emerged: Optional[int]
    dose_trace: list[float] = field(default_factory=list)
    load_trace: list[float] = field(default_factory=list)
    resistance_trace: list[float] = field(default_factory=list)
    in_distribution: bool = True  # False = OOD evaluation episode


@dataclass
class EvalResults:
    """Aggregated results across many episodes."""
    policy_name: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    clearance_rate: float
    mean_clearance_day: float
    mean_final_resistance: float
    ood_gap: float = 0.0        # (in-dist reward - OOD reward)
    records: list[EpisodeRecord] = field(default_factory=list)


def evaluate_policy(
    policy,
    env,
    n_episodes: int = 200,
    resistance_profiles: Optional[list[dict]] = None,
    policy_name: str = "policy",
    rng: Optional[np.random.Generator] = None,
) -> EvalResults:
    """
    Evaluate a policy over multiple episodes.

    Parameters
    ----------
    policy             : object with .predict(obs) -> action interface
    env                : AMREnv instance
    n_episodes         : number of episodes to run
    resistance_profiles: list of dicts with 'profile_id' and 'initial_resistance'
                         If provided, cycles through profiles for OOD testing.
    """
    rng = rng or np.random.default_rng(42)
    records = []

    for ep_idx in range(n_episodes):
        if hasattr(policy, "reset"):
            policy.reset()

        # Optionally set specific resistance profile
        options: dict = {}
        profile_id = "default"
        if resistance_profiles:
            profile = resistance_profiles[ep_idx % len(resistance_profiles)]
            profile_id = profile.get("profile_id", str(ep_idx))
            options["random_init_resistance"] = False
            options["initial_resistance"] = profile.get("initial_resistance", 0.0)
            if "fitness_cost_override" in profile:
                options["fitness_cost_override"] = profile["fitness_cost_override"]
            if "ec50_multiplier" in profile:
                options["ec50_multiplier"] = profile["ec50_multiplier"]

        obs, _ = env.reset(options=options if options else None)

        ep_reward = 0.0
        cleared = False
        clearance_day = None
        day_resistance_emerged = None
        dose_trace, load_trace, resistance_trace = [], [], []
        prev_resistance = env._resistance

        for day in range(env.max_episode_steps):
            action = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            dose_trace.append(info["dose"])
            load_trace.append(info["bacterial_load"])
            resistance_trace.append(info["resistance_level"])

            if info["cleared"] and not cleared:
                cleared = True
                clearance_day = day + 1

            if info["resistance_level"] > prev_resistance and day_resistance_emerged is None:
                day_resistance_emerged = day + 1
            prev_resistance = info["resistance_level"]

            if terminated or truncated:
                break

        records.append(EpisodeRecord(
            policy_name=policy_name,
            resistance_profile_id=profile_id,
            total_reward=ep_reward,
            cleared=cleared,
            clearance_day=clearance_day,
            final_resistance=env._resistance,
            final_load=env._load,
            day_resistance_emerged=day_resistance_emerged,
            dose_trace=dose_trace,
            load_trace=load_trace,
            resistance_trace=resistance_trace,
        ))

    rewards = [r.total_reward for r in records]
    clearances = [r.cleared for r in records]
    clearance_days = [r.clearance_day for r in records if r.clearance_day is not None]
    final_resistances = [r.final_resistance for r in records]

    return EvalResults(
        policy_name=policy_name,
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        clearance_rate=float(np.mean(clearances)),
        mean_clearance_day=float(np.mean(clearance_days)) if clearance_days else float(env.max_episode_steps),
        mean_final_resistance=float(np.mean(final_resistances)),
        records=records,
    )


def compute_ood_gap(in_dist_results: EvalResults, ood_results: EvalResults) -> float:
    """OOD generalisation gap = in-distribution reward - OOD reward."""
    return in_dist_results.mean_reward - ood_results.mean_reward


def time_to_resistance_analysis(records: list[EpisodeRecord]) -> dict:
    """
    Kaplan-Meier survival analysis for time-to-resistance emergence.

    Returns KM curve data (for plotting) and median TTR.
    Requires lifelines.
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        log.warning("lifelines not installed. Skipping survival analysis.")
        return {}

    durations = []
    events = []  # 1 = resistance emerged, 0 = censored (never emerged)

    for r in records:
        if r.day_resistance_emerged is not None:
            durations.append(r.day_resistance_emerged)
            events.append(1)
        else:
            # Censored at episode end
            durations.append(len(r.resistance_trace))
            events.append(0)

    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events)

    return {
        "median_ttr": kmf.median_survival_time_,
        "timeline": kmf.timeline.tolist(),
        "km_estimate": kmf.survival_function_["KM_estimate"].tolist(),
        "ci_lower": kmf.confidence_interval_["KM_estimate_lower_0.95"].tolist(),
        "ci_upper": kmf.confidence_interval_["KM_estimate_upper_0.95"].tolist(),
    }


def compare_policies(results_dict: dict[str, EvalResults]) -> pd.DataFrame:
    """Build a comparison table across policies."""
    rows = []
    for name, r in results_dict.items():
        rows.append({
            "policy": name,
            "mean_reward": f"{r.mean_reward:.3f} ± {r.std_reward:.3f}",
            "clearance_rate": f"{r.clearance_rate:.1%}",
            "mean_clearance_day": f"{r.mean_clearance_day:.1f}",
            "mean_final_resistance": f"{r.mean_final_resistance:.2f}",
            "ood_gap": f"{r.ood_gap:.3f}" if r.ood_gap else "—",
        })
    return pd.DataFrame(rows).set_index("policy")


def generate_ood_profiles(
    n_profiles: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Generate held-out resistance profiles for OOD evaluation.
    These should NOT appear in training episodes.
    """
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(n_profiles):
        profiles.append({
            "profile_id": f"ood_{i:03d}",
            "initial_resistance": float(rng.choice([1, 2, 3])),
            "fitness_cost_override": float(rng.uniform(0.04, 0.16)),
            "ec50_multiplier": float(rng.uniform(1.5, 4.0)),
        })
    return profiles
