"""
scripts/evaluate.py

Full evaluation run — loads trained checkpoints and generates all paper results.

Usage:
  python scripts/evaluate.py \
      --policy_path checkpoints/best/best_model.zip \
      --fixed_policy_path checkpoints/fixed_ppo.zip \
      --adversary_path checkpoints/adversary_final.pt \
      --output_dir results/

Outputs:
  results/
    metrics_table.csv        — Table 1 in paper
    fig1_learning_curves.pdf
    fig2_km_survival.pdf
    fig3_ood_gap.pdf
    fig4_load_traces.pdf
    fig5_resistance_heatmap.pdf
    eval_records.json        — Full episode records for inspection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config_utils import load_config as _load_config_cast, _cast

from simulator.envs.amr_env import AMREnv
from resistance.model.resistance_model import AdversarialResistanceModel
from baselines.baselines import CyclingHeuristic, ContextualBanditPolicy, MaxDosePolicy, NoDosePolicy
from evaluation.metrics.eval_metrics import (
    evaluate_policy, compute_ood_gap, time_to_resistance_analysis,
    compare_policies, generate_ood_profiles, EvalResults,
)
from evaluation.plots.paper_figures import (
    plot_learning_curves, plot_km_survival, plot_ood_bar,
    plot_load_traces, plot_resistance_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_sb3_policy(path: str):
    from stable_baselines3 import PPO
    path = str(path).removesuffix('.zip')

    class _SB3Wrapper:
        def __init__(self, model):
            self.model = model
        def predict(self, obs, deterministic=True):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action)
        def reset(self):
            pass

    return _SB3Wrapper(PPO.load(path))


def make_env(cfg: dict, resistance_model=None) -> AMREnv:
    return AMREnv(
        drug=cfg["env"]["drug"],
        pathogen=cfg["env"]["pathogen"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
        bacterial_load_init=cfg["env"]["bacterial_load_init"],
        target_load=float(cfg["env"]["target_load"]),
        fitness_cost_slope=cfg["resistance"]["fitness_cost_slope"],
        resistance_model=resistance_model,
        seed=0,
    )


def run_all_evaluations(
    policies: dict,
    env: AMREnv,
    ood_profiles: list[dict],
    n_episodes: int,
) -> tuple[dict[str, EvalResults], dict[str, EvalResults]]:
    """Returns (in_dist_results, ood_results)."""
    in_dist, ood = {}, {}

    for name, policy in policies.items():
        log.info(f"Evaluating {name} (in-distribution)...")
        in_dist[name] = evaluate_policy(
            policy, env, n_episodes=n_episodes, policy_name=name
        )

        log.info(f"Evaluating {name} (OOD)...")
        ood[name] = evaluate_policy(
            policy, env, n_episodes=n_episodes,
            resistance_profiles=ood_profiles,
            policy_name=name,
        )
        in_dist[name].ood_gap = compute_ood_gap(in_dist[name], ood[name])

    return in_dist, ood


def build_ood_bar_data(
    in_dist: dict[str, EvalResults],
    ood_results: dict[str, EvalResults],
) -> dict:
    return {
        name: {
            "in_dist":     in_dist[name].mean_reward,
            "ood":         ood_results[name].mean_reward,
            "in_dist_std": in_dist[name].std_reward,
            "ood_std":     ood_results[name].std_reward,
        }
        for name in in_dist
    }


def build_km_data(
    in_dist: dict[str, EvalResults],
) -> dict:
    km = {}
    for name, results in in_dist.items():
        km_result = time_to_resistance_analysis(results.records)
        if km_result:
            km[name] = km_result
    return km


def build_load_traces(
    in_dist: dict[str, EvalResults],
    n_traces: int = 50,
) -> dict:
    traces = {}
    for name, results in in_dist.items():
        traces[name] = [r.load_trace for r in results.records[:n_traces]]
    return traces


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AMR policies")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to adversarial PPO checkpoint (.zip)")
    parser.add_argument("--fixed_policy_path", type=str,
                        default="checkpoints/fixed_ppo_static.zip",
                        help="Path to fixed-resistance PPO checkpoint (.zip)")
    parser.add_argument("--adversary_path", type=str, default=None,
                        help="Path to trained adversary checkpoint (.pt)")
    parser.add_argument("--training_history", type=str, default=None,
                        help="Path to adversarial PPO training_history.json for learning curves")
    parser.add_argument("--fixed_training_history", type=str, default=None,
                        help="Path to fixed-resistance PPO training_history.json for learning curves overlay")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_ood_profiles", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config_safe(args.config)
    env = make_env(cfg)

    # ── Build policy dict ─────────────────────────────────────────────────
    policies = {}

    log.info(f"Loading adversarial PPO from {args.policy_path}")
    policies["adversarial_ppo"] = load_sb3_policy(args.policy_path)

    if args.fixed_policy_path:
        fp = Path(args.fixed_policy_path)
        # SB3 appends .zip automatically; check both forms
        if not fp.exists() and not fp.with_suffix("").with_suffix(".zip").exists():
            fp_zip = Path(str(fp) + ".zip")
            if not fp_zip.exists():
                log.warning(f"fixed_policy_path not found: {args.fixed_policy_path} — skipping")
                args.fixed_policy_path = None
        if args.fixed_policy_path:
            log.info(f"Loading fixed-resistance PPO from {args.fixed_policy_path}")
            policies["fixed_ppo"] = load_sb3_policy(args.fixed_policy_path)

    policies["cycling"] = CyclingHeuristic(period=cfg["baselines"]["cycling_period"])
    policies["bandit"] = ContextualBanditPolicy(epsilon=cfg["baselines"]["bandit_epsilon"])
    policies["max_dose"] = MaxDosePolicy()

    # ── OOD profiles ─────────────────────────────────────────────────────
    ood_profiles = generate_ood_profiles(n_profiles=args.n_ood_profiles, seed=99)

    # ── Run evaluations ───────────────────────────────────────────────────
    in_dist, ood_results = run_all_evaluations(
        policies, env, ood_profiles, n_episodes=args.n_episodes
    )

    # ── Metrics table ─────────────────────────────────────────────────────
    table = compare_policies({**in_dist})
    table_path = out / "metrics_table.csv"
    table.to_csv(table_path)
    log.info(f"\n{table.to_string()}")
    log.info(f"Metrics table saved to {table_path}")

    # ── Figures ───────────────────────────────────────────────────────────
    fig_dir = out / "figures"

    # Fig 1: Learning curves — overlay all available policy histories
    learning_histories: dict = {}
    if args.training_history:
        with open(args.training_history) as f:
            learning_histories["adversarial_ppo"] = json.load(f)
    if args.fixed_training_history:
        with open(args.fixed_training_history) as f:
            learning_histories["fixed_ppo"] = json.load(f)
    if learning_histories:
        plot_learning_curves(
            learning_histories,
            output_path=str(fig_dir / "fig1_learning_curves.pdf"),
        )

    # Fig 2: KM survival
    km_data = build_km_data(in_dist)
    if km_data:
        plot_km_survival(km_data, output_path=str(fig_dir / "fig2_km_survival.pdf"))

    # Fig 3: OOD bar
    ood_bar_data = build_ood_bar_data(in_dist, ood_results)
    plot_ood_bar(ood_bar_data, output_path=str(fig_dir / "fig3_ood_gap.pdf"))

    # Fig 4: Load traces
    load_traces = build_load_traces(in_dist)
    plot_load_traces(
        load_traces,
        output_path=str(fig_dir / "fig4_load_traces.pdf"),
        target_load=float(cfg["env"]["target_load"]),
    )

    # Fig 5: Resistance heatmap
    _generate_resistance_heatmap(env, cfg, fig_dir)

    # ── Save raw records ──────────────────────────────────────────────────
    records_out = []
    for name, results in in_dist.items():
        for r in results.records:
            records_out.append({
                "policy": name,
                "reward": r.total_reward,
                "cleared": r.cleared,
                "clearance_day": r.clearance_day,
                "final_resistance": r.final_resistance,
                "profile_id": r.resistance_profile_id,
            })
    with open(out / "eval_records.json", "w") as f:
        json.dump(records_out, f, indent=2)

    log.info(f"\nAll outputs saved to {out}/")


def _generate_resistance_heatmap(env: AMREnv, cfg: dict, fig_dir: Path) -> None:
    """Quick simulation to build dose×day resistance grid."""
    from simulator.envs.amr_env import DOSE_LEVELS

    n_days = cfg["env"]["max_episode_steps"]
    n_doses = len(DOSE_LEVELS)
    resistance_grid = np.zeros((n_days, n_doses))

    rng = np.random.default_rng(0)
    for d_idx, dose in enumerate(DOSE_LEVELS):
        env.reset()
        for day in range(n_days):
            obs, _, term, trunc, info = env.step(d_idx)
            resistance_grid[day, d_idx] = info["resistance_level"]
            if term or trunc:
                break

    dose_grid = np.tile(DOSE_LEVELS, (n_days, 1))
    plot_resistance_heatmap(
        dose_grid, resistance_grid,
        output_path=str(fig_dir / "fig5_resistance_heatmap.pdf"),
    )


def load_config_safe(path: str) -> dict:
    try:
        return _load_config_cast(path)
    except FileNotFoundError:
        log.warning(f"Config not found at {path}, using defaults")
        return _cast({
            "env": {"drug": "ciprofloxacin", "pathogen": "e_coli",
                    "max_episode_steps": 14, "bacterial_load_init": 1e8, "target_load": 1e3},
            "resistance": {"fitness_cost_slope": 0.08},
            "baselines": {"cycling_period": 3, "bandit_epsilon": 0.1},
        })


if __name__ == "__main__":
    main()
