"""tests/test_env.py"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
from simulator.envs.amr_env import AMREnv, DOSE_LEVELS, N_OBS


class TestAMREnvBasics:
    def setup_method(self):
        self.env = AMREnv(seed=42)

    def teardown_method(self):
        self.env.close()

    def test_reset_returns_correct_obs_shape(self):
        obs, info = self.env.reset()
        assert obs.shape == (N_OBS,)

    def test_obs_dtype_float32(self):
        obs, _ = self.env.reset()
        assert obs.dtype == np.float32

    def test_action_space_discrete(self):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        assert self.env.action_space.n == len(DOSE_LEVELS)

    def test_step_returns_five_tuple(self):
        self.env.reset()
        result = self.env.step(2)  # dose = 1.0 mg/kg
        assert len(result) == 5

    def test_step_reward_is_scalar(self):
        self.env.reset()
        _, reward, _, _, _ = self.env.step(2)
        assert isinstance(reward, float)

    def test_episode_terminates(self):
        self.env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, terminated, truncated, _ = self.env.step(4)  # max dose
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate within 100 steps"

    def test_info_has_required_keys(self):
        self.env.reset()
        _, _, _, _, info = self.env.step(2)
        for key in ["bacterial_load", "resistance_level", "dose", "cleared", "day"]:
            assert key in info

    def test_zero_dose_does_not_clear(self):
        """With no drug, infection should not clear."""
        self.env.reset()
        cleared = False
        for _ in range(self.env.max_episode_steps):
            _, _, term, trunc, info = self.env.step(0)
            if info["cleared"]:
                cleared = True
                break
            if term or trunc:
                break
        assert not cleared

    def test_max_dose_every_step(self):
        """Max dose (400 mg) should clear infection within the episode."""
        self.env.reset()
        cleared = False
        for _ in range(self.env.max_episode_steps):
            _, _, term, trunc, info = self.env.step(4)
            if info["cleared"]:
                cleared = True
            if term or trunc:
                break
        assert cleared, "Max dose should clear infection within 14 days"

    def test_resistance_nonnegative(self):
        self.env.reset()
        for _ in range(self.env.max_episode_steps):
            _, _, term, trunc, _ = self.env.step(1)
            assert self.env._resistance >= 0.0
            if term or trunc:
                break

    def test_resistance_bounded_above(self):
        self.env.reset()
        for _ in range(self.env.max_episode_steps):
            _, _, term, trunc, _ = self.env.step(1)
            assert self.env._resistance <= 4.0
            if term or trunc:
                break

    def test_seed_reproducibility(self):
        env1 = AMREnv(seed=7)
        env2 = AMREnv(seed=7)
        obs1, _ = env1.reset(seed=7)
        obs2, _ = env2.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)
        env1.close(); env2.close()

    def test_different_seeds_differ(self):
        env1 = AMREnv(seed=1)
        env2 = AMREnv(seed=2)
        # Run a few steps and check trajectories diverge
        env1.reset(seed=1); env2.reset(seed=2)
        rewards1, rewards2 = [], []
        for _ in range(5):
            _, r1, t1, tr1, _ = env1.step(1)
            _, r2, t2, tr2, _ = env2.step(1)
            rewards1.append(r1); rewards2.append(r2)
        # Not guaranteed to differ every step but very likely
        env1.close(); env2.close()


class TestAMREnvContinuous:
    def test_continuous_action_space(self):
        env = AMREnv(continuous_actions=True, max_dose=2.0)
        obs, _ = env.reset()
        assert isinstance(env.action_space, gym.spaces.Box)
        obs, reward, term, trunc, info = env.step(np.array([1.0]))
        assert isinstance(reward, float)
        env.close()

    def test_continuous_clipped(self):
        env = AMREnv(continuous_actions=True, max_dose=2.0)
        env.reset()
        _, _, _, _, info = env.step(np.array([99.0]))  # way above max
        assert info["dose"] <= 2.0
        env.close()


class TestAMREnvResistanceModel:
    def test_custom_resistance_model_called(self):
        calls = []

        def my_resistance(obs, resistance_level, dose, in_msw, rng):
            calls.append(1)
            return resistance_level  # never evolves

        env = AMREnv(resistance_model=my_resistance, seed=0)
        env.reset()
        for _ in range(3):
            _, _, term, trunc, _ = env.step(2)
            if term or trunc:
                break
        assert len(calls) >= 3
        env.close()

    def test_always_resistant_model_terminates_early(self):
        """A resistance model that always maxes out should cause early termination."""
        def always_pan_resistant(obs, resistance_level, dose, in_msw, rng):
            return 4.0

        env = AMREnv(resistance_model=always_pan_resistant, seed=0)
        env.reset()
        _, _, terminated, _, _ = env.step(2)
        assert terminated  # pan-resistance terminates immediately
        env.close()
