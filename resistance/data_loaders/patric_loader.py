"""
resistance/data_loaders/patric_loader.py

Data pipeline for BV-BRC / CARD genotype-MIC data.

Two loaders:

  MockPATRICLoader   — synthetic data for dev/testing (no download required)
  PATRICLoader       — real data from download_data.py output files

PATRICLoader joins AMR phenotype records with per-genome specialty gene
annotations to build a binary gene presence/absence feature matrix.

If the inner join yields fewer than `min_joined` isolates (e.g., the gene
file covers only a small fraction of sequenced isolates), the loader falls
back to MIC-binned synthetic features so pretraining can proceed without
crashing.

File formats (output of scripts/download_data.py)
--------------------------------------------------
  amr_tsv   : TSV with columns
                genome_id, genome_name, antibiotic, resistant_phenotype,
                measurement_value, measurement_unit, measurement_sign
  gene_tsv  : TSV with columns
                genome_id, gene, product, classification, antibiotic

EUCAST breakpoints for ciprofloxacin (mg/L)
-------------------------------------------
  Susceptible : MIC <= 0.25
  Resistant   : MIC >  0.50
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


# ── Gene list ─────────────────────────────────────────────────────────────────

ECOLI_CIPRO_GENES: list[str] = [
    "gyrA_S83L", "gyrA_D87N", "parC_S80I", "parC_E84V",
    "qnrA", "qnrB", "qnrC", "qnrD", "qnrS",
    "aac(6')-Ib-cr", "oqxA", "oqxB",
    "marR", "soxR", "tolC", "acrB",
]

N_GENES = len(ECOLI_CIPRO_GENES)   # 16

EUCAST_SUSCEPTIBLE = 0.25   # mg/L  (S <= this)
EUCAST_RESISTANT   = 0.50   # mg/L  (R > this)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AMRDataset(Dataset):
    """Torch Dataset of (genotype_features, log_mic) pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── Mock loader ───────────────────────────────────────────────────────────────

class MockPATRICLoader:
    """
    Synthetic genotype-MIC data for development without downloaded data.

    Models a realistic genotype-phenotype relationship:
      log(MIC) ~ N(mu, sigma) where mu depends on number of resistance genes.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def load(
        self,
        antibiotic: str = "ciprofloxacin",
        n: int = 2000,
        val_fraction: float = 0.15,
    ) -> tuple[DataLoader, DataLoader]:
        X = self.rng.integers(0, 2, size=(n, N_GENES)).astype(np.float32)
        gene_effects = self.rng.uniform(0.5, 2.0, size=N_GENES)
        log_mic = X @ gene_effects + self.rng.normal(0, 0.3, size=n)
        log_mic = np.clip(log_mic, 0, 12)

        n_val = int(n * val_fraction)
        train_ds = AMRDataset(X[n_val:], log_mic[n_val:])
        val_ds   = AMRDataset(X[:n_val], log_mic[:n_val])
        return DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(val_ds, batch_size=64)

    def gene_names(self) -> list[str]:
        return ECOLI_CIPRO_GENES.copy()


# ── Real loader ───────────────────────────────────────────────────────────────

class PATRICLoader:
    """
    Real BV-BRC AMR data loader.

    Parameters
    ----------
    amr_tsv   : path to patric_ecoli_cipro.tsv  (from download_data.py)
    gene_tsv  : path to card_cipro_genes.txt    (from download_data.py)
    """

    def __init__(self, amr_tsv: str, gene_tsv: str):
        self.amr_tsv  = Path(amr_tsv)
        self.gene_tsv = Path(gene_tsv)
        self._amr_df:   Optional[pd.DataFrame] = None
        self._gene_df:  Optional[pd.DataFrame] = None
        self._pivot:    Optional[pd.DataFrame] = None
        self._gene_cols: Optional[list[str]]   = None

    # ── Raw loaders ──────────────────────────────────────────────────────────

    def _load_amr(self) -> pd.DataFrame:
        if self._amr_df is None:
            self._amr_df = pd.read_csv(self.amr_tsv, sep="\t", low_memory=False)
        return self._amr_df

    def _load_gene_pivot(self) -> pd.DataFrame:
        """Return genome_id × gene binary presence/absence DataFrame."""
        if self._pivot is None:
            gene_df = pd.read_csv(self.gene_tsv, sep="\t", low_memory=False)
            self._gene_df = gene_df
            if gene_df.empty or "gene" not in gene_df.columns:
                self._pivot = pd.DataFrame()
                self._gene_cols = []
            else:
                # crosstab: rows = genome_id, cols = gene names, values = count
                pivot = pd.crosstab(gene_df["genome_id"], gene_df["gene"])
                pivot = (pivot > 0).astype(np.float32)
                self._pivot    = pivot
                self._gene_cols = pivot.columns.tolist()
        return self._pivot

    # ── Public API ────────────────────────────────────────────────────────────

    def gene_names(self) -> list[str]:
        """Unique gene names present in the gene annotation file."""
        self._load_gene_pivot()
        return list(self._gene_cols or [])

    def load(
        self,
        antibiotic: str = "ciprofloxacin",
        val_fraction: float = 0.15,
        min_joined: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Build (train_loader, val_loader) from joined AMR + gene data.

        Falls back to MIC-binned synthetic features if the inner join
        yields fewer than `min_joined` isolates.
        """
        rng = rng or np.random.default_rng(42)
        amr_df = self._load_amr()
        pivot  = self._load_gene_pivot()

        # Filter to target antibiotic
        sub = amr_df[amr_df["antibiotic"].str.lower() == antibiotic.lower()].copy()
        sub = sub.dropna(subset=["measurement_value"])
        sub["measurement_value"] = pd.to_numeric(sub["measurement_value"], errors="coerce")
        sub = sub.dropna(subset=["measurement_value"])

        joined: Optional[pd.DataFrame] = None
        if not pivot.empty and "genome_id" in sub.columns:
            joined = sub.merge(pivot, on="genome_id", how="inner")

        if joined is None or len(joined) < min_joined:
            n_joined = 0 if joined is None else len(joined)
            log.warning(
                f"Inner join of AMR records with gene data yielded only {n_joined} isolates "
                f"(< min_joined={min_joined}). "
                "Falling back to MIC-binned synthetic gene features. "
                "Run scripts/download_data.py to get more gene annotations."
            )
            return self._fallback_load(sub, val_fraction, rng)

        gene_cols = [c for c in self._gene_cols if c in joined.columns]
        X = joined[gene_cols].fillna(0).values.astype(np.float32)
        y = np.log1p(joined["measurement_value"].values.astype(np.float32))

        return self._make_loaders(X, y, val_fraction, rng)

    def data_summary(self) -> dict:
        """
        Return summary statistics over the AMR TSV.

        Keys: n_isolates, mic_median, mic_q1, mic_q3,
              n_susceptible, n_intermediate, n_resistant,
              class_balance (R / total), n_unique_genes
        """
        amr_df = self._load_amr()
        pivot  = self._load_gene_pivot()

        mics = pd.to_numeric(amr_df["measurement_value"], errors="coerce").dropna()
        n_s  = int((mics <= EUCAST_SUSCEPTIBLE).sum())
        n_r  = int((mics >  EUCAST_RESISTANT).sum())
        n_i  = len(mics) - n_s - n_r

        return {
            "n_isolates":     len(amr_df),
            "mic_median":     float(mics.median()),
            "mic_q1":         float(mics.quantile(0.25)),
            "mic_q3":         float(mics.quantile(0.75)),
            "n_susceptible":  n_s,
            "n_intermediate": n_i,
            "n_resistant":    n_r,
            "class_balance":  round(n_r / max(len(mics), 1), 4),
            "n_unique_genes": len(self._gene_cols or []),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fallback_load(
        self,
        sub: pd.DataFrame,
        val_fraction: float,
        rng: np.random.Generator,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Synthetic features from MIC bins when the gene join is too sparse.
        Creates N_GENES binary features with additive MIC signal so the
        MIC head can still learn a genotype-phenotype mapping.
        """
        mics = sub["measurement_value"].values.astype(np.float32)
        n = len(mics)

        if n == 0:
            # Nothing to work with — return empty loaders
            dummy_X = np.zeros((2, N_GENES), dtype=np.float32)
            dummy_y = np.zeros(2, dtype=np.float32)
            return self._make_loaders(dummy_X, dummy_y, val_fraction, rng)

        # Bin MICs into resistance tiers (0=S, 1=I, 2=R) and build
        # correlated binary gene profiles: more resistant → more genes present.
        bins = np.zeros(n, dtype=int)
        bins[mics > EUCAST_SUSCEPTIBLE] = 1
        bins[mics > EUCAST_RESISTANT]   = 2

        X = np.zeros((n, N_GENES), dtype=np.float32)
        for i, tier in enumerate(bins):
            n_present = rng.integers(0, tier * 3 + 1)  # 0,1,2 → 0–1, 0–4, 0–7 genes
            n_present = int(np.clip(n_present, 0, N_GENES))
            idx = rng.choice(N_GENES, size=n_present, replace=False)
            X[i, idx] = 1.0
        X += rng.normal(0, 0.05, size=X.shape).astype(np.float32)
        X = np.clip(X, 0, 1)

        y = np.log1p(mics)
        return self._make_loaders(X, y, val_fraction, rng)

    @staticmethod
    def _make_loaders(
        X: np.ndarray,
        y: np.ndarray,
        val_fraction: float,
        rng: np.random.Generator,
    ) -> tuple[DataLoader, DataLoader]:
        n   = len(X)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        train_ds = AMRDataset(X[idx[n_val:]], y[idx[n_val:]])
        val_ds   = AMRDataset(X[idx[:n_val]], y[idx[:n_val]])
        return (
            DataLoader(train_ds, batch_size=64, shuffle=True),
            DataLoader(val_ds,   batch_size=64),
        )
