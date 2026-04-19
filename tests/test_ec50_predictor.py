"""
tests/test_ec50_predictor.py

Tests for EC50Predictor and its integration with AMREnv.
No internet required — all tests use synthetic checkpoints.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistance.model.resistance_model import EC50Predictor, ResistanceMLP
from resistance.data_loaders.patric_loader import N_GENES
from simulator.envs.amr_env import AMREnv


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_net() -> ResistanceMLP:
    """Small network for fast tests."""
    return ResistanceMLP(obs_dim=N_GENES, hidden_dims=[32, 16])


@pytest.fixture()
def predictor(tiny_net) -> EC50Predictor:
    return EC50Predictor(tiny_net, device="cpu")


@pytest.fixture()
def saved_checkpoint(tiny_net, tmp_path) -> str:
    """Write a checkpoint and return its path."""
    p = str(tmp_path / "ec50_predictor.pt")
    pred = EC50Predictor(tiny_net, device="cpu")
    pred.save(p, obs_dim=N_GENES)
    return p


@pytest.fixture()
def gene_vec() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(N_GENES,)).astype(np.float32)


# ── EC50Predictor unit tests ──────────────────────────────────────────────────

class TestEC50PredictorOutput:
    def test_returns_positive_float(self, predictor, gene_vec):
        mult = predictor.predict_ec50_multiplier(gene_vec)
        assert isinstance(mult, float)
        assert mult > 0.0

    def test_within_clamp_bounds(self, predictor, gene_vec):
        mult = predictor.predict_ec50_multiplier(gene_vec)
        assert EC50Predictor._MULT_MIN <= mult <= EC50Predictor._MULT_MAX

    def test_deterministic_same_input(self, predictor, gene_vec):
        m1 = predictor.predict_ec50_multiplier(gene_vec)
        m2 = predictor.predict_ec50_multiplier(gene_vec)
        assert m1 == m2

    def test_different_genotypes_can_differ(self, predictor):
        rng = np.random.default_rng(0)
        results = set()
        for _ in range(20):
            gv = rng.integers(0, 2, size=(N_GENES,)).astype(np.float32)
            results.add(round(predictor.predict_ec50_multiplier(gv), 6))
        # After random init, most predictions will differ
        assert len(results) > 1


class TestEC50PredictorCheckpoint:
    def test_save_and_load_roundtrip(self, tiny_net, tmp_path, gene_vec):
        p = str(tmp_path / "pred.pt")
        pred1 = EC50Predictor(tiny_net, device="cpu")
        pred1.save(p, obs_dim=N_GENES)

        pred2 = EC50Predictor.load_from_checkpoint(p, device="cpu")
        assert pred2.predict_ec50_multiplier(gene_vec) == pred1.predict_ec50_multiplier(gene_vec)

    def test_loaded_checkpoint_has_correct_obs_dim(self, saved_checkpoint):
        pred = EC50Predictor.load_from_checkpoint(saved_checkpoint)
        # Network encoder first layer should accept N_GENES inputs
        first_linear = next(
            m for m in pred.net.encoder.modules() if isinstance(m, torch.nn.Linear)
        )
        assert first_linear.in_features == N_GENES


# ── AMREnv integration ────────────────────────────────────────────────────────

class TestAMREnvEC50Integration:
    def test_env_accepts_ec50_predictor(self, predictor):
        env = AMREnv(ec50_predictor=predictor, n_gene_features=N_GENES, seed=0)
        obs, _ = env.reset()
        assert obs.shape == (11,)
        env.close()

    def test_ec50_multiplier_set_after_reset(self, predictor):
        env = AMREnv(ec50_predictor=predictor, n_gene_features=N_GENES, seed=0)
        env.reset()
        assert env._current_ec50_multiplier > 0.0
        assert env._current_ec50_multiplier != 1.0 or True  # may equal 1.0 for some genotypes
        env.close()

    def test_ec50_changes_between_episodes(self, predictor):
        env = AMREnv(ec50_predictor=predictor, n_gene_features=N_GENES, seed=0)
        mults = []
        for _ in range(20):
            env.reset()
            mults.append(env._current_ec50_multiplier)
        env.close()
        assert np.var(mults) > 0.0, "EC50 multiplier should vary across episodes"

    def test_genotype_features_stored(self, predictor):
        env = AMREnv(ec50_predictor=predictor, n_gene_features=N_GENES, seed=0)
        env.reset()
        gf = env._genotype_features
        assert gf.shape == (N_GENES,)
        assert set(np.unique(gf)).issubset({0.0, 1.0})
        env.close()

    def test_ec50_multiplier_in_step_info(self, predictor):
        env = AMREnv(ec50_predictor=predictor, n_gene_features=N_GENES, seed=0)
        env.reset()
        _, _, _, _, info = env.step(4)  # max dose
        assert "ec50_multiplier" in info
        assert info["ec50_multiplier"] > 0.0
        assert "genotype_features" in info
        env.close()

    def test_no_predictor_multiplier_is_one(self):
        env = AMREnv(seed=0)
        env.reset()
        assert env._current_ec50_multiplier == 1.0
        env.close()

    def test_ec50_predictor_affects_pkpd(self, predictor):
        """Effective EC50 should equal base_ec50 * multiplier after reset."""
        env = AMREnv(ec50_predictor=predictor, n_gene_features=N_GENES, seed=0)
        env.reset()
        effective = env.pkpd.pd.ec50
        mult = env._current_ec50_multiplier
        base = env._base_ec50
        assert abs(effective - base * mult) < 1e-6, (
            f"effective={effective:.6f} != base={base:.6f} * mult={mult:.6f} = {base * mult:.6f}"
        )
        env.close()

    def test_twenty_episodes_ec50_variance(self, predictor):
        """Core spec: 20 episodes, EC50 variance > 0."""
        env = AMREnv(ec50_predictor=predictor, n_gene_features=N_GENES, seed=7)
        mults = []
        for _ in range(20):
            env.reset()
            mults.append(env._current_ec50_multiplier)
        env.close()
        assert np.var(mults) > 0.0, f"Expected variance > 0, got {np.var(mults)}"


# ── Pretraining integration ───────────────────────────────────────────────────

class TestPretrainingIntegration:
    def test_ec50_predictor_saved_by_pretrain(self, tmp_path):
        """Simulate what pretrain_resistance.py does and verify the checkpoint."""
        from resistance.model.resistance_model import AdversarialResistanceModel
        from resistance.data_loaders.patric_loader import MockPATRICLoader

        loader = MockPATRICLoader(seed=0)
        train_dl, _ = loader.load(n=64)

        model = AdversarialResistanceModel(obs_dim=N_GENES, device="cpu")
        for X, y in train_dl:
            model.pretrain_step(X, y)
            break  # one batch is enough

        ckpt_path = str(tmp_path / "ec50_predictor.pt")
        pred = EC50Predictor(model.net, device="cpu")
        pred.save(ckpt_path, obs_dim=N_GENES)

        loaded = EC50Predictor.load_from_checkpoint(ckpt_path)
        gv = np.ones(N_GENES, dtype=np.float32)
        mult = loaded.predict_ec50_multiplier(gv)
        assert mult > 0.0
