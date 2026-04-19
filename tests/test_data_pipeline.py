"""
tests/test_data_pipeline.py

Data pipeline tests — all fixtures are synthetic (no internet required).
Covers PATRICLoader loading, feature matrix shape/values, data_summary(),
and the sparse-join fallback path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resistance.data_loaders.patric_loader import (
    PATRICLoader,
    MockPATRICLoader,
    ECOLI_CIPRO_GENES,
    N_GENES,
    EUCAST_SUSCEPTIBLE,
    EUCAST_RESISTANT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_amr(tmp_path) -> Path:
    """Synthetic AMR TSV matching the real BV-BRC genome_amr schema."""
    rng = np.random.default_rng(0)
    n = 200
    mic_ladder = [0.015, 0.03, 0.06, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    mic_values = rng.choice(mic_ladder, size=n)
    phenotypes = [
        "Susceptible" if m <= EUCAST_SUSCEPTIBLE
        else ("Intermediate" if m <= EUCAST_RESISTANT else "Resistant")
        for m in mic_values
    ]
    df = pd.DataFrame({
        "genome_id":          [f"111.{i}" for i in range(n)],
        "genome_name":        [f"Escherichia coli {i}" for i in range(n)],
        "antibiotic":         ["ciprofloxacin"] * n,
        "resistant_phenotype": phenotypes,
        "measurement_value":  mic_values,
        "measurement_unit":   ["mg/L"] * n,
        "measurement_sign":   ["="] * n,
    })
    p = tmp_path / "patric_ecoli_cipro.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p


@pytest.fixture()
def tmp_gene(tmp_path, tmp_amr) -> Path:
    """Synthetic gene TSV matching the real BV-BRC sp_gene schema."""
    amr_df = pd.read_csv(tmp_amr, sep="\t")
    rng = np.random.default_rng(1)
    rows = []
    # Give ~80% of isolates gene annotations so the join is dense
    genome_ids = amr_df["genome_id"].tolist()
    for gid in rng.choice(genome_ids, size=int(len(genome_ids) * 0.8), replace=False):
        n_genes = int(np.clip(rng.poisson(2), 1, 5))
        for gene in rng.choice(ECOLI_CIPRO_GENES, size=n_genes, replace=False):
            rows.append({
                "genome_id":      gid,
                "gene":           gene,
                "product":        f"{gene} protein",
                "classification": "fluoroquinolone resistance",
                "antibiotic":     "ciprofloxacin",
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "card_cipro_genes.txt"
    df.to_csv(p, sep="\t", index=False)
    return p


@pytest.fixture()
def tmp_gene_sparse(tmp_path, tmp_amr) -> Path:
    """Gene TSV covering only 10 isolates — triggers fallback in PATRICLoader."""
    amr_df = pd.read_csv(tmp_amr, sep="\t")
    genome_ids = amr_df["genome_id"].tolist()[:10]
    rows = [
        {"genome_id": gid, "gene": ECOLI_CIPRO_GENES[0],
         "product": "gyrA", "classification": "fluoro", "antibiotic": "cipro"}
        for gid in genome_ids
    ]
    p = tmp_path / "card_cipro_genes_sparse.txt"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPATRICLoaderLoad:
    def test_returns_two_dataloaders(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        train_dl, val_dl = loader.load()
        assert train_dl is not None
        assert val_dl is not None

    def test_feature_matrix_shape(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        train_dl, val_dl = loader.load()
        # Collect all batches from train and val
        X_batches = [X for X, _ in train_dl] + [X for X, _ in val_dl]
        assert len(X_batches) > 0
        n_cols = X_batches[0].shape[1]
        # Columns = number of unique genes in the gene file
        assert n_cols >= 1
        assert n_cols <= N_GENES

    def test_feature_matrix_binary(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        train_dl, _ = loader.load()
        for X, _ in train_dl:
            arr = X.numpy()
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0

    def test_labels_are_finite(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        train_dl, val_dl = loader.load()
        for _, y in list(train_dl) + list(val_dl):
            assert np.isfinite(y.numpy()).all()

    def test_gene_names_nonempty(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        names = loader.gene_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(g, str) for g in names)


class TestDataSummary:
    def test_runs_without_error(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        summary = loader.data_summary()
        assert isinstance(summary, dict)

    def test_required_keys(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        summary = loader.data_summary()
        for key in ("n_isolates", "mic_median", "mic_q1", "mic_q3",
                    "n_susceptible", "n_resistant", "class_balance", "n_unique_genes"):
            assert key in summary, f"Missing key: {key}"

    def test_class_balance_in_range(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        summary = loader.data_summary()
        assert 0.0 <= summary["class_balance"] <= 1.0

    def test_mic_statistics_sensible(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        summary = loader.data_summary()
        assert summary["mic_q1"] <= summary["mic_median"] <= summary["mic_q3"]
        assert summary["n_susceptible"] + summary["n_intermediate"] + summary["n_resistant"] \
               == summary["n_isolates"]

    def test_n_isolates_matches_tsv(self, tmp_amr, tmp_gene):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene))
        summary = loader.data_summary()
        raw = pd.read_csv(tmp_amr, sep="\t")
        assert summary["n_isolates"] == len(raw)


class TestFallback:
    def test_fallback_triggers_on_sparse_join(self, tmp_amr, tmp_gene_sparse, caplog):
        import logging
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene_sparse))
        with caplog.at_level(logging.WARNING):
            train_dl, val_dl = loader.load(min_joined=50)
        assert any("Falling back" in r.message for r in caplog.records)

    def test_fallback_loaders_work(self, tmp_amr, tmp_gene_sparse):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene_sparse))
        train_dl, val_dl = loader.load(min_joined=50)
        # Should still return usable data
        X_list = [X for X, _ in train_dl]
        assert len(X_list) > 0
        assert X_list[0].shape[1] == N_GENES  # fallback always uses N_GENES columns

    def test_fallback_features_in_range(self, tmp_amr, tmp_gene_sparse):
        loader = PATRICLoader(str(tmp_amr), str(tmp_gene_sparse))
        train_dl, _ = loader.load(min_joined=50)
        for X, _ in train_dl:
            arr = X.numpy()
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0


class TestMockPATRICLoader:
    def test_loads_synthetic_data(self):
        loader = MockPATRICLoader(seed=0)
        train_dl, val_dl = loader.load(n=128)
        assert len(list(train_dl)) > 0

    def test_gene_names_returns_list(self):
        loader = MockPATRICLoader()
        names = loader.gene_names()
        assert names == ECOLI_CIPRO_GENES
