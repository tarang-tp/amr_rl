"""
scripts/train.py

Main entry point for adversarial co-training.

Usage:
  # Full run with defaults
  python scripts/train.py

  # Override config values
  python scripts/train.py --config config.yaml \
      --total_timesteps 500000 \
      --seed 7 \
      --device cuda \
      --use_wandb

  # Load pretrained resistance model
  python scripts/train.py \
      --pretrained_adversary checkpoints/resistance_pretrained.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
import torch

# TF32 gives free throughput on Ampere/Blackwell GPUs with no meaningful
# precision loss for RL workloads.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Ensure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.adversarial.co_trainer import AdversarialCoTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_trainer(cfg: dict, args: argparse.Namespace) -> AdversarialCoTrainer:
    ec = cfg["experiment"]
    env_cfg = cfg["env"]
    pk_cfg = cfg["pkpd"]
    rew_cfg = cfg["reward"]
    pol_cfg = cfg["policy"]
    adv_cfg = cfg["adversarial"]
    res_cfg = cfg["resistance"]
    eval_cfg = cfg["eval"]

    # Environment kwargs passed to AMREnv
    env_kwargs = {
        "drug":               env_cfg["drug"],
        "pathogen":           env_cfg["pathogen"],
        "max_episode_steps":  env_cfg["max_episode_steps"],
        "bacterial_load_init": env_cfg["bacterial_load_init"],
        "target_load":        env_cfg["target_load"],
        "fitness_cost_slope": res_cfg["fitness_cost_slope"],
        "seed":               args.seed,
    }

    # PPO kwargs (SB3 naming)
    policy_kwargs = {
        "learning_rate":  pol_cfg["learning_rate"],
        "n_steps":        pol_cfg["n_steps"],
        "batch_size":     pol_cfg["batch_size"],
        "n_epochs":       pol_cfg["n_epochs"],
        "gamma":          pol_cfg["gamma"],
        "gae_lambda":     pol_cfg["gae_lambda"],
        "clip_range":     pol_cfg["clip_range"],
        "ent_coef":       pol_cfg["ent_coef"],
        "vf_coef":        pol_cfg["vf_coef"],
        "max_grad_norm":  pol_cfg["max_grad_norm"],
        "policy_kwargs":  {"net_arch": pol_cfg["net_arch"]},
    }

    # Adversary kwargs
    adversary_kwargs = {
        "obs_dim":      14,   # 11 obs + resistance + dose + in_msw
        "hidden_dims":  res_cfg["hidden_dims"],
        "dropout":      res_cfg["dropout"],
        "lr":           res_cfg["adversarial_lr"],
    }

    total_ts = args.total_timesteps or pol_cfg["total_timesteps"]

    return AdversarialCoTrainer(
        env_kwargs=env_kwargs,
        policy_kwargs=policy_kwargs,
        adversary_kwargs=adversary_kwargs,
        total_timesteps=total_ts,
        co_train_ratio=adv_cfg["co_train_ratio"],
        adversary_update_freq=res_cfg["adversarial_update_freq"],
        log_dir=ec["log_dir"],
        checkpoint_dir=ec["checkpoint_dir"],
        eval_freq=eval_cfg["checkpoint_freq"],
        n_eval_episodes=eval_cfg["n_eval_episodes"] // 10,  # fast in-loop eval
        seed=args.seed,
        device=args.device,
        use_wandb=args.use_wandb,
        ec50_predictor_path=args.ec50_predictor,
    )


def main():
    parser = argparse.ArgumentParser(description="Adversarial co-training for AMR treatment")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--pretrained_adversary", type=str, default=None,
                        help="Path to pretrained resistance model checkpoint")
    parser.add_argument("--ec50_predictor", type=str, default=None,
                        help="Path to standalone EC50 predictor checkpoint (.pt)")
    parser.add_argument("--resume_policy", type=str, default=None,
                        help="Path to existing policy checkpoint to resume from")
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    log.info(f"Loaded config from {args.config}")
    log.info(f"Experiment: {cfg['experiment']['name']}")

    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="amr_rl",
                name=cfg["experiment"]["name"],
                config=cfg,
            )
            log.info("W&B initialised")
        except ImportError:
            log.warning("wandb not installed, skipping")
            args.use_wandb = False

    trainer = build_trainer(cfg, args)

    # Optionally load pretrained adversary
    if args.pretrained_adversary:
        trainer.load_pretrained_adversary(args.pretrained_adversary)

    # Optionally resume policy
    if args.resume_policy:
        from stable_baselines3 import PPO
        trainer.policy = PPO.load(args.resume_policy, env=trainer.train_env)
        log.info(f"Resumed policy from {args.resume_policy}")

    # Run
    history = trainer.train()

    # Save history
    import json
    hist_path = Path(cfg["experiment"]["log_dir"]) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved to {hist_path}")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
