"""
simulator/pkpd/pharmacokinetics.py

One-compartment PK model with oral/IV dosing and a Hill-equation PD kill-rate.

Bacterial dynamics follow:
    dN/dt = (mu_eff - kill_rate) * N

where:
    mu_eff    = base_growth_rate * (1 - fitness_cost * resistance_level)
    kill_rate = E_max * C^h / (EC50^h + C^h)
    C(t)      = C_peak * exp(-ke * t)   (one-compartment, first-order elimination)

All concentrations in mg/L, time in hours (internally), days at env boundary.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PKParams:
    """Pharmacokinetic parameters for a single antibiotic."""
    name: str = "ciprofloxacin"
    volume_of_distribution: float = 2.5    # L/kg
    half_life_hours: float = 4.0
    bioavailability: float = 0.85
    protein_binding: float = 0.30          # fraction bound (unbound is active)
    dose_unit: str = "mg/kg"

    @property
    def elimination_rate(self) -> float:
        """ke = ln(2) / t_half  (h^-1)"""
        return np.log(2) / self.half_life_hours

    def peak_concentration(self, dose: float, body_weight_kg: float = 70.0) -> float:
        """Cmax (mg/L) after a single oral dose."""
        return (dose * self.bioavailability * (1.0 - self.protein_binding)) / (
            self.volume_of_distribution * body_weight_kg
        )

    def concentration_at(self, dose: float, t_hours: float,
                          body_weight_kg: float = 70.0) -> float:
        """C(t) for a single-dose bolus (mg/L), t in hours post-dose."""
        c_peak = self.peak_concentration(dose, body_weight_kg)
        return c_peak * np.exp(-self.elimination_rate * t_hours)

    def auc_24(self, dose: float, body_weight_kg: float = 70.0) -> float:
        """AUC_0-24 (mg·h/L) — analytic integral of C(t) over 24h."""
        c_peak = self.peak_concentration(dose, body_weight_kg)
        # integral of C_peak * exp(-ke*t) from 0 to 24
        return c_peak / self.elimination_rate * (1 - np.exp(-self.elimination_rate * 24))


@dataclass
class PDParams:
    """Pharmacodynamic Hill-equation kill model (E_max model)."""
    e_max: float = 1.0           # maximal kill rate (h^-1)
    ec50: float = 0.5            # concentration achieving 50% E_max (mg/L) — MIC proxy
    hill_coefficient: float = 1.5

    def kill_rate(self, concentration: float) -> float:
        """E(C) = E_max * C^h / (EC50^h + C^h)  — vectorised."""
        c = np.maximum(concentration, 0.0)
        ch = c ** self.hill_coefficient
        ec50h = self.ec50 ** self.hill_coefficient
        return self.e_max * ch / (ec50h + ch)

    def mic(self, mu_eff: float) -> float:
        """Concentration at which kill_rate == mu_eff (static MIC analogue)."""
        # Solve: e_max * c^h / (ec50^h + c^h) = mu_eff
        # => c^h = mu_eff * ec50^h / (e_max - mu_eff)
        if mu_eff >= self.e_max:
            return np.inf
        ratio = mu_eff / (self.e_max - mu_eff)
        return self.ec50 * (ratio ** (1.0 / self.hill_coefficient))

    def mutant_selection_window(self, mu_eff: float, mic_mutant: float) -> tuple[float, float]:
        """
        MSW = (MIC_wildtype, MIC_mutant).
        Dosing within this window selects for resistant mutants.
        Returns (lower, upper) concentration bounds.
        """
        mic_wt = self.mic(mu_eff)
        return (mic_wt, mic_mutant)


class PKPDModel:
    """
    Integrates PK and PD to simulate bacterial dynamics over a dosing episode.

    Internally uses hourly Euler integration; exposes a daily step interface.
    """

    HOURS_PER_DAY = 24

    def __init__(
        self,
        pk: PKParams,
        pd: PDParams,
        base_growth_rate: float = 0.5,      # h^-1 at susceptible state
        body_weight_kg: float = 70.0,
        dt_hours: float = 0.5,              # integration step
        rng: Optional[np.random.Generator] = None,
    ):
        self.pk = pk
        self.pd = pd
        self.base_growth_rate = base_growth_rate
        self.body_weight_kg = body_weight_kg
        self.dt = dt_hours
        self.rng = rng or np.random.default_rng()

        self._concentration: float = 0.0   # current plasma [drug] mg/L
        self._last_dose_time: float = 0.0  # hours since last dose
        self._elapsed_hours: float = 0.0

    def reset(self) -> None:
        self._concentration = 0.0
        self._last_dose_time = 0.0
        self._elapsed_hours = 0.0

    def administer_dose(self, dose: float) -> None:
        """Add a bolus dose (mg/kg) — adds to current plasma concentration."""
        c_peak = self.pk.peak_concentration(dose, self.body_weight_kg)
        self._concentration += c_peak
        self._last_dose_time = self._elapsed_hours

    def step_day(
        self,
        dose: float,
        bacterial_load: float,
        resistance_level: float,
        fitness_cost_slope: float = 0.08,
    ) -> tuple[float, float, dict]:
        """
        Simulate one day (24h) given a dose.

        Parameters
        ----------
        dose              : daily dose in mg/kg
        bacterial_load    : N(t) in CFU/mL at start of day
        resistance_level  : float in [0,1], modulates growth and kill rate
        fitness_cost_slope: how much each resistance unit suppresses growth

        Returns
        -------
        (new_load, mean_concentration, info_dict)
        """
        self.administer_dose(dose)

        n = bacterial_load
        mean_c = 0.0
        steps = int(self.HOURS_PER_DAY / self.dt)
        conc_trace = []

        for _ in range(steps):
            # PK: exponential decay
            self._concentration *= np.exp(-self.pk.elimination_rate * self.dt)
            self._elapsed_hours += self.dt

            c = max(self._concentration, 0.0)
            conc_trace.append(c)

            # Effective growth rate (fitness cost of resistance)
            mu_eff = self.base_growth_rate * (1.0 - fitness_cost_slope * resistance_level)
            mu_eff = max(mu_eff, 0.0)

            # PD kill rate (resistance scales EC50 upward — harder to kill)
            ec50_adj = self.pd.ec50 * (1.0 + resistance_level * 2.0)
            ch = c ** self.pd.hill_coefficient
            ec50h = ec50_adj ** self.pd.hill_coefficient
            kill = self.pd.e_max * ch / (ec50h + ch + 1e-12)

            # Euler step on bacterial load
            dN = (mu_eff - kill) * n * self.dt
            n = max(n + dN, 0.0)

        mean_c = float(np.mean(conc_trace))
        peak_c = float(np.max(conc_trace))

        # MSW check: are we dosing in the selection window?
        mic_wt = self.pd.mic(self.base_growth_rate)
        mic_resistant = self.pd.mic(
            self.base_growth_rate * (1.0 - fitness_cost_slope * min(resistance_level + 1, 4))
        )
        in_msw = mic_wt < mean_c < mic_resistant

        info = {
            "mean_concentration": mean_c,
            "peak_concentration": peak_c,
            "auc_24": float(np.sum(conc_trace) * self.dt),
            "in_mutant_selection_window": in_msw,
            "mic_wildtype": mic_wt,
        }

        return float(n), mean_c, info
