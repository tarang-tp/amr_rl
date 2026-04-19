"""
simulator/envs/amr_env.py

Gymnasium environment for antibiotic treatment under evolving resistance.

Observation space  (11-dim Box):
  [log10_bacterial_load, resistance_level, drug_concentration,
   day_of_treatment, days_remaining,
   cum_dose, last_dose,
   in_msw, auc_24, mic_wt, peak_concentration]

Action space (Discrete or Box depending on config):
  Discrete: dose levels {0, 0.5, 1.0, 1.5, 2.0} mg/kg  (default)
  Box:      continuous dose in [0, max_dose]
"""

from __future__ import annotations

import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any

from simulator.pkpd.pharmacokinetics import PKParams, PDParams, PKPDModel
from simulator.reward.reward_fn import RewardFunction, RewardConfig


# Pre-defined drug profiles
DRUG_PROFILES: dict[str, tuple[PKParams, PDParams]] = {
    "ciprofloxacin": (
        PKParams(name="ciprofloxacin", volume_of_distribution=2.5,
                 half_life_hours=4.0, bioavailability=0.85, protein_binding=0.30),
        PDParams(e_max=1.0, ec50=0.1, hill_coefficient=1.5),   # cipro MIC90 ~0.06 mg/L
    ),
    "meropenem": (
        PKParams(name="meropenem", volume_of_distribution=0.35,
                 half_life_hours=1.0, bioavailability=1.0, protein_binding=0.02),
        PDParams(e_max=0.8, ec50=0.06, hill_coefficient=2.0),  # meropenem MIC90 ~0.06 mg/L
    ),
    "vancomycin": (
        PKParams(name="vancomycin", volume_of_distribution=0.7,
                 half_life_hours=6.0, bioavailability=1.0, protein_binding=0.55),
        PDParams(e_max=0.6, ec50=0.5, hill_coefficient=1.2),   # vancomycin MIC90 ~1 mg/L (body_weight adjusted)
    ),
}

DOSE_LEVELS = np.array([0.0, 100.0, 200.0, 300.0, 400.0], dtype=np.float32)  # mg (total dose)
N_OBS = 11


class AMREnv(gym.Env):
    """
    Single-pathogen antibiotic treatment environment.

    The resistance model is injected at construction time — during adversarial
    co-training the training loop replaces it each episode with a freshly
    updated adversarial resistance model.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        drug: str = "ciprofloxacin",
        pathogen: str = "e_coli",
        max_episode_steps: int = 14,
        bacterial_load_init: float = 1e8,
        target_load: float = 1e3,
        fitness_cost_slope: float = 0.08,
        resistance_model=None,             # callable(obs, rng) -> new resistance level
        reward_config: Optional[RewardConfig] = None,
        continuous_actions: bool = False,
        max_dose: float = 400.0,
        seed: Optional[int] = None,
        ec50_predictor=None,               # callable(genotype_features) -> float (EC50 multiplier)
        n_gene_features: int = 16,         # dimension of genotype feature vector
    ):
        super().__init__()
        self.drug = drug
        self.pathogen = pathogen
        self.max_episode_steps = int(max_episode_steps)
        self.bacterial_load_init = float(bacterial_load_init)
        self.target_load = float(target_load)
        self.fitness_cost_slope = fitness_cost_slope
        self.resistance_model = resistance_model
        self.continuous_actions = continuous_actions
        self.max_dose = max_dose
        self.ec50_predictor = ec50_predictor
        self.n_gene_features = n_gene_features

        # Copy so each env instance has independent PK/PD parameters
        pk, pd = DRUG_PROFILES[drug]
        self.pkpd = PKPDModel(pk=copy.copy(pk), pd=copy.copy(pd), rng=np.random.default_rng(seed))
        self._base_fitness_cost_slope = fitness_cost_slope
        self._base_ec50 = self.pkpd.pd.ec50
        self.reward_fn = RewardFunction(reward_config or RewardConfig(
            target_load=self.target_load, initial_load=self.bacterial_load_init
        ))

        # Action space
        if continuous_actions:
            self.action_space = spaces.Box(
                low=0.0, high=max_dose, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(len(DOSE_LEVELS))

        # Observation space — all normalised to ~[0,1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32
        )

        # State
        self._load: float = bacterial_load_init
        self._resistance: float = 0.0      # 0=susceptible, 4=pan-resistant
        self._day: int = 0
        self._cum_dose: float = 0.0
        self._last_dose: float = 0.0
        self._last_info: dict = {}
        self._genotype_features: np.ndarray = np.zeros(n_gene_features, dtype=np.float32)
        self._current_ec50_multiplier: float = 1.0

        self._rng = np.random.default_rng(seed)

    # ── Gym interface ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._load = float(self.bacterial_load_init)
        self._resistance = 0.0
        self._day = 0
        self._cum_dose = 0.0
        self._last_dose = 0.0
        self._last_info = {}
        self.pkpd.reset()

        # Restore base PK/PD params and genotype state
        self.fitness_cost_slope = self._base_fitness_cost_slope
        self.pkpd.pd.ec50 = self._base_ec50
        self._genotype_features = np.zeros(self.n_gene_features, dtype=np.float32)
        self._current_ec50_multiplier = 1.0

        # Sample episode genotype and predict EC50 multiplier (if predictor provided)
        if self.ec50_predictor is not None:
            genotype = self._rng.integers(0, 2, size=(self.n_gene_features,)).astype(np.float32)
            self._genotype_features = genotype
            multiplier = float(self.ec50_predictor(genotype))
            self._current_ec50_multiplier = multiplier
            self.pkpd.pd.ec50 = self._base_ec50 * multiplier

        if options:
            if options.get("random_init_resistance", False):
                self._resistance = float(self._rng.integers(0, 3))
            if "initial_resistance" in options:
                self._resistance = float(options["initial_resistance"])
            if "fitness_cost_override" in options:
                self.fitness_cost_slope = float(options["fitness_cost_override"])
            if "ec50_multiplier" in options:
                self.pkpd.pd.ec50 = self._base_ec50 * float(options["ec50_multiplier"])

        return self._get_obs(), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert not self._is_done(), "Episode already terminated. Call reset()."

        # Decode action
        if self.continuous_actions:
            dose = float(np.clip(action[0], 0.0, self.max_dose))
        else:
            dose = float(DOSE_LEVELS[int(action)])

        prev_load = self._load
        prev_resistance = self._resistance

        # PK/PD step
        new_load, mean_c, pkpd_info = self.pkpd.step_day(
            dose=dose,
            bacterial_load=self._load,
            resistance_level=self._resistance,
            fitness_cost_slope=self.fitness_cost_slope,
        )

        # Resistance evolution
        new_resistance = self._evolve_resistance(
            obs=self._get_obs(), dose=dose, in_msw=pkpd_info["in_mutant_selection_window"]
        )

        self._load = new_load
        self._resistance = new_resistance
        self._day += 1
        self._cum_dose += dose
        self._last_dose = dose

        terminated = self._is_done()
        truncated = self._day >= self.max_episode_steps and not terminated

        reward, rew_info = self.reward_fn(
            bacterial_load=self._load,
            prev_load=prev_load,
            dose=dose,
            resistance_level=self._resistance,
            prev_resistance_level=prev_resistance,
            in_msw=pkpd_info["in_mutant_selection_window"],
            done=terminated or truncated,
        )

        info = {**pkpd_info, **rew_info,
                "day": self._day,
                "bacterial_load": self._load,
                "resistance_level": self._resistance,
                "dose": dose,
                "cleared": self._load <= self.target_load,
                "ec50_multiplier": self._current_ec50_multiplier,
                "genotype_features": self._genotype_features.copy()}

        self._last_info = info
        return self._get_obs(), reward, terminated, truncated, info

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        load_norm = np.log10(max(self._load, 1.0)) / 10.0
        c = self.pkpd._concentration
        return np.array([
            load_norm,
            self._resistance / 4.0,
            c / 5.0,
            self._day / self.max_episode_steps,
            (self.max_episode_steps - self._day) / self.max_episode_steps,
            self._cum_dose / (self.max_dose * self.max_episode_steps),
            self._last_dose / self.max_dose,
            float(self._last_info.get("in_mutant_selection_window", 0)),
            self._last_info.get("auc_24", 0.0) / 50.0,
            self._last_info.get("mic_wildtype", 0.5) / 2.0,
            self._last_info.get("peak_concentration", 0.0) / 5.0,
        ], dtype=np.float32)

    def _evolve_resistance(self, obs: np.ndarray, dose: float, in_msw: bool) -> float:
        """
        Evolve resistance level using injected resistance_model or default Markov model.
        """
        if self.resistance_model is not None:
            return float(self.resistance_model(obs, self._resistance, dose, in_msw, self._rng))
        return self._default_markov_resistance(dose, in_msw)

    def _default_markov_resistance(self, dose: float, in_msw: bool) -> float:
        """
        Simple baseline Markov resistance model.
        Transition probability increases if dose is in MSW or sub-therapeutic.
        """
        base_prob = 0.02
        msw_boost = 0.10 if in_msw else 0.0
        # Sub-therapeutic dosing also selects
        sub_therapeutic_boost = 0.05 if dose < 0.5 else 0.0
        p_increase = min(base_prob + msw_boost + sub_therapeutic_boost, 0.3)
        p_decrease = 0.005  # very rare reversion

        if self._resistance < 4.0 and self._rng.random() < p_increase:
            return min(self._resistance + 1.0, 4.0)
        elif self._resistance > 0.0 and self._rng.random() < p_decrease:
            return max(self._resistance - 1.0, 0.0)
        return self._resistance

    def _is_done(self) -> bool:
        return (
            self._load <= self.target_load
            or self._day >= self.max_episode_steps
            or self._resistance >= 4.0  # pan-resistance = treatment failure
        )

    def render(self) -> None:
        print(
            f"Day {self._day:02d} | "
            f"Load: {self._load:.2e} CFU/mL | "
            f"Resistance: {self._resistance:.1f} | "
            f"[Drug]: {self.pkpd._concentration:.3f} mg/L"
        )
