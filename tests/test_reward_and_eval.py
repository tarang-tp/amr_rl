"""tests/test_reward_and_eval.py"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.reward.reward_fn import RewardFunction, RewardConfig
from evaluation.metrics.eval_metrics import (
    EpisodeRecord, EvalResults, compute_ood_gap, generate_ood_profiles,
)


# ── Reward function ───────────────────────────────────────────────────────────

class TestRewardFunction:
    def setup_method(self):
        self.rf = RewardFunction(RewardConfig(
            w_clearance=5.0, w_load=-0.01, w_dose=-0.005,
            w_resistance=-2.0, msw_penalty=-1.5,
            target_load=1e3, initial_load=1e8,
        ))

    def _call(self, **kwargs):
        defaults = dict(
            bacterial_load=1e6, prev_load=1e7, dose=1.0,
            resistance_level=0.0, prev_resistance_level=0.0,
            in_msw=False, done=False,
        )
        defaults.update(kwargs)
        return self.rf(**defaults)

    def test_returns_float_and_dict(self):
        r, info = self._call()
        assert isinstance(r, float)
        assert isinstance(info, dict)

    def test_info_has_all_components(self):
        _, info = self._call()
        for key in ["load", "dose", "resistance", "msw", "clearance", "total"]:
            assert key in info

    def test_clearance_bonus_on_done_and_cleared(self):
        r, info = self._call(bacterial_load=500.0, done=True)
        assert info["clearance"] == pytest.approx(5.0)

    def test_no_clearance_bonus_if_not_done(self):
        _, info = self._call(bacterial_load=500.0, done=False)
        assert info["clearance"] == 0.0

    def test_no_clearance_bonus_if_load_too_high(self):
        _, info = self._call(bacterial_load=1e6, done=True)
        assert info["clearance"] == 0.0

    def test_msw_penalty_when_in_msw(self):
        _, info_msw = self._call(in_msw=True)
        _, info_no = self._call(in_msw=False)
        assert info_msw["msw"] < info_no["msw"]

    def test_resistance_penalty_on_increase(self):
        _, info = self._call(resistance_level=2.0, prev_resistance_level=1.0)
        assert info["resistance"] < 0.0

    def test_no_resistance_penalty_if_stable(self):
        _, info = self._call(resistance_level=1.0, prev_resistance_level=1.0)
        assert info["resistance"] == 0.0

    def test_dose_penalty_positive_dose(self):
        _, info = self._call(dose=1.0)
        assert info["dose"] < 0.0

    def test_zero_dose_no_dose_penalty(self):
        _, info = self._call(dose=0.0)
        assert info["dose"] == 0.0

    def test_total_matches_sum_of_components(self):
        r, info = self._call(in_msw=True, dose=1.5, bacterial_load=1e5)
        component_sum = info["load"] + info["dose"] + info["resistance"] + info["msw"] + info["clearance"]
        assert abs(r - component_sum) < 1e-9


# ── Eval metrics ──────────────────────────────────────────────────────────────

class TestOODGap:
    def _make_results(self, name, mean_rew):
        return EvalResults(
            policy_name=name,
            n_episodes=10,
            mean_reward=mean_rew,
            std_reward=0.1,
            clearance_rate=0.5,
            mean_clearance_day=7.0,
            mean_final_resistance=1.0,
        )

    def test_ood_gap_positive_when_ood_worse(self):
        in_dist = self._make_results("p", 5.0)
        ood = self._make_results("p", 3.0)
        assert compute_ood_gap(in_dist, ood) == pytest.approx(2.0)

    def test_ood_gap_zero_when_equal(self):
        in_dist = self._make_results("p", 4.0)
        ood = self._make_results("p", 4.0)
        assert compute_ood_gap(in_dist, ood) == pytest.approx(0.0)


class TestOODProfiles:
    def test_generates_correct_count(self):
        profiles = generate_ood_profiles(n_profiles=20)
        assert len(profiles) == 20

    def test_profile_has_required_keys(self):
        profiles = generate_ood_profiles(n_profiles=5)
        for p in profiles:
            assert "profile_id" in p
            assert "initial_resistance" in p

    def test_initial_resistance_in_valid_range(self):
        profiles = generate_ood_profiles(n_profiles=30)
        for p in profiles:
            assert 0 <= p["initial_resistance"] <= 4

    def test_reproducible_with_seed(self):
        p1 = generate_ood_profiles(n_profiles=10, seed=42)
        p2 = generate_ood_profiles(n_profiles=10, seed=42)
        assert p1[0]["initial_resistance"] == p2[0]["initial_resistance"]

    def test_different_seeds_differ(self):
        p1 = generate_ood_profiles(n_profiles=10, seed=1)
        p2 = generate_ood_profiles(n_profiles=10, seed=2)
        # Profiles should differ
        assert any(
            p1[i]["initial_resistance"] != p2[i]["initial_resistance"]
            for i in range(10)
        )
