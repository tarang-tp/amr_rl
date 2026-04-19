# amr_rl

Adversarially co-trained RL for antimicrobial resistance treatment.

## Structure

```
amr_rl/
├── config.yaml                        # Master config (all hyperparameters)
├── simulator/
│   ├── envs/amr_env.py                # Gymnasium environment
│   ├── pkpd/pharmacokinetics.py       # PK/PD model (one-compartment + Hill)
│   └── reward/reward_fn.py            # Reward with MSW penalty
├── resistance/
│   ├── model/resistance_model.py      # Adversarial neural resistance model
│   ├── data_loaders/patric_loader.py  # PATRIC/CARD data pipeline
│   └── pretraining/pretrain_resistance.py
├── training/
│   └── adversarial/co_trainer.py      # Minimax co-training loop (PPO vs adversary)
├── baselines/
│   └── baselines.py                   # Cycling, bandit, max-dose
└── evaluation/
    └── metrics/eval_metrics.py        # OOD gap, KM survival, comparison table
```

## Quickstart

```bash
pip install -e ".[dev]"

# Pretrain resistance model on synthetic data
python -m resistance.pretraining.pretrain_resistance --mock --epochs 30

# (Coming) Full adversarial co-training
python scripts/train.py --config config.yaml
```

## Coming next
- `evaluation/plots/` — paper figures
- `scripts/train.py` + `scripts/evaluate.py`
- `tests/`
- `notebooks/`
