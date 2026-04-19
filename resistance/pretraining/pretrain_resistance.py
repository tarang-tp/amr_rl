"""
resistance/pretraining/pretrain_resistance.py

Pretrain the resistance model's MIC-prediction head on PATRIC/CARD data
before adversarial co-training begins.

Usage:
  python -m resistance.pretraining.pretrain_resistance \
      --mock              # use synthetic data (no download needed)
      --epochs 50 \
      --output checkpoints/resistance_pretrained.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from resistance.model.resistance_model import AdversarialResistanceModel, EC50Predictor
from resistance.data_loaders.patric_loader import MockPATRICLoader, PATRICLoader, N_GENES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def pretrain(
    model: AdversarialResistanceModel,
    train_loader,
    val_loader,
    n_epochs: int = 50,
    patience: int = 10,
) -> dict:
    best_val = np.inf
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.net.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            loss = model.pretrain_step(X_batch, y_batch)
            train_losses.append(loss)

        # ── Validate ──
        model.net.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X = X_batch.to(model.device)
                y = y_batch.to(model.device).float()
                _, mic_pred = model.net(X)
                import torch.nn.functional as F
                loss = F.mse_loss(torch.log1p(mic_pred), torch.log1p(y))
                val_losses.append(float(loss.item()))

        train_mean = np.mean(train_losses)
        val_mean = np.mean(val_losses)
        history["train_loss"].append(train_mean)
        history["val_loss"].append(val_mean)

        log.info(f"Epoch {epoch:03d}/{n_epochs} | train_loss={train_mean:.4f} val_loss={val_mean:.4f}")

        if val_mean < best_val - 1e-4:
            best_val = val_mean
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use synthetic data")
    parser.add_argument("--amr_tsv",   type=str, default="data/processed/patric_ecoli_cipro.tsv")
    parser.add_argument("--gene_tsv",  type=str, default="data/processed/card_cipro_genes.txt")
    parser.add_argument("--antibiotic", type=str, default="ciprofloxacin")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="checkpoints/resistance_pretrained.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.mock:
        log.info("Using MockPATRICLoader (synthetic data)")
        loader = MockPATRICLoader()
        obs_dim = N_GENES   # gene features only — no obs prefix during pretraining
        train_loader, val_loader = loader.load(n=2000)
    else:
        log.info(f"Loading BV-BRC data from {args.amr_tsv}")
        loader = PATRICLoader(args.amr_tsv, args.gene_tsv)
        train_loader, val_loader = loader.load(args.antibiotic)
        n_genes = len(loader.gene_names())
        obs_dim = n_genes if n_genes > 0 else N_GENES
        if n_genes > 0:
            summary = loader.data_summary()
            log.info(
                f"Data: {summary['n_isolates']} isolates, "
                f"MIC median={summary['mic_median']:.3f} mg/L, "
                f"{summary['n_unique_genes']} genes"
            )

    model = AdversarialResistanceModel(obs_dim=obs_dim, lr=args.lr, device=args.device)

    log.info("Starting pretraining...")
    history = pretrain(model, train_loader, val_loader, n_epochs=args.epochs)

    model.save(args.output)
    log.info(f"Saved pretrained resistance model to {args.output}")

    # Save standalone EC50 predictor checkpoint (MIC head only)
    ec50_path = Path(args.output).with_name("ec50_predictor.pt")
    predictor = EC50Predictor(model.net, device=args.device)
    predictor.save(str(ec50_path), obs_dim=obs_dim)
    log.info(f"Saved standalone EC50 predictor to {ec50_path}")

    return history


if __name__ == "__main__":
    main()
