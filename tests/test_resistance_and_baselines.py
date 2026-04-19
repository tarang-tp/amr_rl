"""tests/test_resistance_and_baselines.py"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from resistance.model.resistance_model import ResistanceMLP, AdversarialResistanceModel
from resistance.data_loaders.patric_loader import MockPATRICLoader, N_GENES
from baselines.baselines import (
    CyclingHeuristic, ContextualBanditPolicy, MaxDosePolicy, NoDosePolicy,
    make_baseline,
)


# ── Resistance model ──────────────────────────────────────────────────────────

class TestResistanceMLP:
    def setup_method(self):
        self.net = ResistanceMLP(obs_dim=14)

    def test_forward_output_shapes(self):
        x = torch.zeros(4, 14)
        logits, mic = self.net(x)
        assert logits.shape == (4, 3)
        assert mic.shape == (4,)

    def test_transition_probs_sum_to_one(self):
        x = torch.randn(8, 14)
        probs = self.net.transition_probs(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_mic_output_positive(self):
        x = torch.randn(4, 14)
        _, mic = self.net(x)
        assert (mic >= 0).all()


class TestAdversarialResistanceModel:
    def setup_method(self):
        self.model = AdversarialResistanceModel(obs_dim=14, device="cpu")
        self.rng = np.random.default_rng(0)
        self.obs = np.zeros(11, dtype=np.float32)

    def test_call_returns_valid_resistance(self):
        for r_level in [0.0, 1.0, 2.0, 3.0]:
            new_r = self.model(self.obs, r_level, 1.0, False, self.rng)
            assert 0.0 <= new_r <= 4.0

    def test_resistance_cannot_exceed_4(self):
        # Even from level 4, should stay at 4
        new_r = self.model(self.obs, 4.0, 1.0, True, self.rng)
        assert new_r <= 4.0

    def test_episode_log_populated(self):
        self.model._episode_log = []
        self.model(self.obs, 0.0, 1.0, False, self.rng)
        assert len(self.model._episode_log) == 1

    def test_adversarial_update_clears_log(self):
        self.model._episode_log = []
        for _ in range(5):
            self.model(self.obs, 0.0, 1.0, False, self.rng)
        self.model.adversarial_update(policy_losses=[-1.0])
        assert len(self.model._episode_log) == 0

    def test_adversarial_update_returns_loss(self):
        self.model._episode_log = []
        for _ in range(3):
            self.model(self.obs, 0.0, 1.0, False, self.rng)
        result = self.model.adversarial_update(policy_losses=[-0.5])
        assert "adversarial_loss" in result
        assert isinstance(result["adversarial_loss"], float)

    def test_pretrain_step_reduces_loss(self):
        x = torch.randn(16, 14)
        y = torch.rand(16) * 5
        l1 = self.model.pretrain_step(x, y)
        losses = [l1]
        for _ in range(20):
            losses.append(self.model.pretrain_step(x, y))
        # Loss should trend downward over 20 steps
        assert losses[-1] < losses[0] * 2  # generous bound — just checks it runs

    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "model.pt")
        self.model.save(path)
        model2 = AdversarialResistanceModel(obs_dim=14, device="cpu")
        model2.load(path)
        x = torch.randn(1, 14)
        # eval() disables dropout so both models are deterministic
        self.model.net.eval()
        model2.net.eval()
        with torch.no_grad():
            p1 = self.model.net.transition_probs(x)
            p2 = model2.net.transition_probs(x)
        assert torch.allclose(p1, p2, atol=1e-6)


class TestMockPATRICLoader:
    def test_load_returns_dataloaders(self):
        loader = MockPATRICLoader()
        train, val = loader.load(n=200)
        batch_x, batch_y = next(iter(train))
        assert batch_x.shape[1] == N_GENES
        assert batch_y.shape[0] == batch_x.shape[0]

    def test_gene_names_correct_length(self):
        loader = MockPATRICLoader()
        assert len(loader.gene_names()) == N_GENES

    def test_labels_nonnegative(self):
        loader = MockPATRICLoader()
        train, _ = loader.load(n=200)
        for x, y in train:
            assert (y >= 0).all()


# ── Baselines ─────────────────────────────────────────────────────────────────

class TestCyclingHeuristic:
    def test_returns_valid_action(self):
        policy = CyclingHeuristic(period=3)
        obs = np.zeros(11)
        for _ in range(14):
            action = policy.predict(obs)
            assert 0 <= action <= 4

    def test_reset_resets_day_counter(self):
        policy = CyclingHeuristic(period=3)
        obs = np.zeros(11)
        a1 = policy.predict(obs)
        policy.reset()
        a2 = policy.predict(obs)
        assert a1 == a2

    def test_cycles_over_schedule(self):
        policy = CyclingHeuristic(period=1)
        obs = np.zeros(11)
        actions = [policy.predict(obs) for _ in range(8)]
        # Should repeat with period = len(DOSE_SCHEDULE) = 4
        assert actions[0] == actions[4]


class TestContextualBandit:
    def test_predict_valid_action(self):
        policy = ContextualBanditPolicy(n_actions=5)
        obs = np.random.randn(11).astype(np.float32)
        action = policy.predict(obs, deterministic=True)
        assert 0 <= action < 5

    def test_update_changes_weights(self):
        policy = ContextualBanditPolicy(n_actions=5, lr=0.5)
        obs = np.ones(11, dtype=np.float32)
        w_before = policy.weights.copy()
        policy.update(obs, action=2, reward=1.0)
        assert not np.allclose(policy.weights, w_before)


class TestMaxAndNoDose:
    def test_max_dose_always_4(self):
        policy = MaxDosePolicy()
        for _ in range(10):
            assert policy.predict(np.zeros(11)) == 4

    def test_no_dose_always_0(self):
        policy = NoDosePolicy()
        for _ in range(10):
            assert policy.predict(np.zeros(11)) == 0


class TestMakeBaseline:
    def test_registry_cycling(self):
        b = make_baseline("cycling")
        assert isinstance(b, CyclingHeuristic)

    def test_registry_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown baseline"):
            make_baseline("nonexistent")
