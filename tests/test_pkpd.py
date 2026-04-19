"""tests/test_pkpd.py"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.pkpd.pharmacokinetics import PKParams, PDParams, PKPDModel


class TestPKParams:
    def test_elimination_rate_positive(self):
        pk = PKParams(half_life_hours=4.0)
        assert pk.elimination_rate > 0

    def test_elimination_rate_formula(self):
        pk = PKParams(half_life_hours=4.0)
        assert abs(pk.elimination_rate - np.log(2) / 4.0) < 1e-10

    def test_peak_concentration_positive(self):
        pk = PKParams()
        assert pk.peak_concentration(dose=10.0) > 0

    def test_peak_concentration_scales_with_dose(self):
        pk = PKParams()
        c1 = pk.peak_concentration(dose=5.0)
        c2 = pk.peak_concentration(dose=10.0)
        assert abs(c2 / c1 - 2.0) < 1e-6

    def test_concentration_decays(self):
        pk = PKParams(half_life_hours=4.0)
        c0 = pk.concentration_at(dose=10.0, t_hours=0.0)
        c_half = pk.concentration_at(dose=10.0, t_hours=4.0)
        assert abs(c_half / c0 - 0.5) < 0.01

    def test_auc_positive(self):
        pk = PKParams()
        assert pk.auc_24(dose=10.0) > 0


class TestPDParams:
    def test_kill_rate_zero_at_zero_concentration(self):
        pd = PDParams()
        assert pd.kill_rate(0.0) == 0.0

    def test_kill_rate_approaches_emax(self):
        pd = PDParams(e_max=1.0)
        assert pd.kill_rate(1000.0) > 0.99

    def test_kill_rate_ec50_gives_half_emax(self):
        pd = PDParams(e_max=1.0, ec50=0.5, hill_coefficient=1.0)
        assert abs(pd.kill_rate(0.5) - 0.5) < 0.01

    def test_mic_infinite_when_mu_exceeds_emax(self):
        pd = PDParams(e_max=0.5)
        assert np.isinf(pd.mic(mu_eff=0.6))

    def test_msw_lower_lt_upper(self):
        pd = PDParams(e_max=1.0, ec50=0.5)
        lo, hi = pd.mutant_selection_window(mu_eff=0.3, mic_mutant=2.0)
        assert lo < hi


class TestPKPDModel:
    def setup_method(self):
        # Use calibrated PDParams matching DRUG_PROFILES in amr_env.py
        self.model = PKPDModel(
            pk=PKParams(),
            pd=PDParams(e_max=1.0, ec50=0.1, hill_coefficient=1.5),
            rng=np.random.default_rng(0),
        )

    def test_step_day_reduces_load_with_high_dose(self):
        # dose=400 mg (calibrated scale) over 3 days should clear a 1e7 load
        self.model.reset()
        load = 1e7
        for _ in range(3):
            load, _, _ = self.model.step_day(
                dose=400.0, bacterial_load=load,
                resistance_level=0.0
            )
        assert load < 1e7

    def test_step_day_load_grows_with_zero_dose(self):
        self.model.reset()
        load = 1e6
        new_load, _, _ = self.model.step_day(
            dose=0.0, bacterial_load=load,
            resistance_level=0.0
        )
        assert new_load > load

    def test_step_day_info_has_required_keys(self):
        self.model.reset()
        _, _, info = self.model.step_day(dose=1.0, bacterial_load=1e6, resistance_level=0.0)
        for key in ["mean_concentration", "peak_concentration", "auc_24",
                    "in_mutant_selection_window", "mic_wildtype"]:
            assert key in info

    def test_step_day_nonnegative_load(self):
        self.model.reset()
        # Very small load + high dose should floor at 0, not go negative
        new_load, _, _ = self.model.step_day(
            dose=2.0, bacterial_load=1.0, resistance_level=0.0
        )
        assert new_load >= 0.0

    def test_concentration_decays_between_days(self):
        self.model.reset()
        self.model.step_day(dose=1.0, bacterial_load=1e6, resistance_level=0.0)
        c1 = self.model._concentration
        self.model.step_day(dose=0.0, bacterial_load=1e6, resistance_level=0.0)
        c2 = self.model._concentration
        assert c2 < c1
