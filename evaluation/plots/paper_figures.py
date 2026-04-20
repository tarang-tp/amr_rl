"""
evaluation/plots/paper_figures.py

Generates all figures for the TMLR paper.

Figures:
  1. learning_curves       — policy reward over training for adversarial vs baselines
  2. km_survival           — Kaplan-Meier time-to-resistance per policy
  3. ood_bar               — in-distribution vs OOD reward gap per policy
  4. bacterial_load_trace  — example episode load trajectories
  5. resistance_heatmap    — resistance emergence rate by dose × day
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "adversarial_ppo": "#534AB7",   # purple — main contribution
    "fixed_ppo":       "#1D9E75",   # teal
    "cycling":         "#D85A30",   # coral
    "bandit":          "#BA7517",   # amber
    "max_dose":        "#888780",   # gray
}

STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "legend.framealpha": 0.85,
}


def _apply_style():
    plt.rcParams.update(STYLE)


# ── Figure 1: Learning curves ─────────────────────────────────────────────────

def plot_learning_curves(
    histories: dict[str, dict],
    output_path: str = "figures/fig1_learning_curves.pdf",
    smooth_window: int = 10,
) -> plt.Figure:
    """
    Parameters
    ----------
    histories : {policy_name: {"timesteps": [...], "policy_reward": [...]}}
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, hist in histories.items():
        ts = np.array(hist["timesteps"]) / 1e6
        rew = np.array(hist["policy_reward"])

        # Smooth
        if len(rew) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(rew, kernel, mode="valid")
            ts_smooth = ts[smooth_window - 1:]
        else:
            smoothed, ts_smooth = rew, ts

        color = PALETTE.get(name, "#333333")
        ax.plot(ts_smooth, smoothed, color=color, linewidth=2,
                label=_pretty_name(name))

        # Shaded std band if available
        if "policy_reward_std" in hist:
            std = np.array(hist["policy_reward_std"])
            if len(std) >= smooth_window:
                std_s = np.convolve(std, kernel, mode="valid")
                ax.fill_between(ts_smooth, smoothed - std_s, smoothed + std_s,
                                color=color, alpha=0.12)

    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Training reward: adversarial vs baselines")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ── Figure 2: Kaplan-Meier survival ──────────────────────────────────────────

def plot_km_survival(
    km_data: dict[str, dict],
    output_path: str = "figures/fig2_km_survival.pdf",
) -> plt.Figure:
    """
    Parameters
    ----------
    km_data : {policy_name: {"timeline": [...], "km_estimate": [...],
                              "ci_lower": [...], "ci_upper": [...]}}
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, data in km_data.items():
        t = np.array(data["timeline"])
        s = np.array(data["km_estimate"])
        color = PALETTE.get(name, "#333333")

        ax.step(t, s, where="post", color=color, linewidth=2,
                label=_pretty_name(name))

        if "ci_lower" in data and "ci_upper" in data:
            lo = np.array(data["ci_lower"])
            hi = np.array(data["ci_upper"])
            ax.fill_between(t, lo, hi, step="post", color=color, alpha=0.12)

    ax.set_xlabel("Day")
    ax.set_ylabel("P(no resistance emergence)")
    ax.set_title("Time-to-resistance: Kaplan-Meier estimates")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ── Figure 3: OOD generalisation gap ─────────────────────────────────────────

def plot_ood_bar(
    results: dict[str, dict],
    output_path: str = "figures/fig3_ood_gap.pdf",
) -> plt.Figure:
    """
    Parameters
    ----------
    results : {policy_name: {"in_dist": float, "ood": float,
                              "in_dist_std": float, "ood_std": float}}
    """
    _apply_style()
    n = len(results)
    x = np.arange(n)
    width = 0.35

    names = list(results.keys())
    in_dist = [results[n_]["in_dist"] for n_ in names]
    ood = [results[n_]["ood"] for n_ in names]
    in_std = [results[n_].get("in_dist_std", 0) for n_ in names]
    ood_std = [results[n_].get("ood_std", 0) for n_ in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_in = ax.bar(x - width / 2, in_dist, width, yerr=in_std,
                     label="In-distribution", color="#534AB7", alpha=0.85,
                     capsize=4, error_kw={"linewidth": 1.2})
    bars_ood = ax.bar(x + width / 2, ood, width, yerr=ood_std,
                      label="Out-of-distribution", color="#AFA9EC", alpha=0.85,
                      capsize=4, error_kw={"linewidth": 1.2})

    # OOD gap annotations
    for i, (iv, ov) in enumerate(zip(in_dist, ood)):
        gap = iv - ov
        if abs(gap) > 0.01:
            y_top = max(iv, ov) + max(in_std[i], ood_std[i]) + 0.05
            ax.annotate(f"Δ{gap:+.2f}", xy=(i, y_top), ha="center",
                        fontsize=9, color="#3C3489")

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_name(n_) for n_ in names], rotation=15, ha="right")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("OOD generalisation: in-distribution vs held-out resistance profiles")
    ax.legend()

    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ── Figure 4: Bacterial load trajectories ────────────────────────────────────

def plot_load_traces(
    traces: dict[str, list[list[float]]],
    output_path: str = "figures/fig4_load_traces.pdf",
    target_load: float = 1e3,
    initial_load: float = 1e8,
) -> plt.Figure:
    """
    Parameters
    ----------
    traces : {policy_name: [[load_day0, load_day1, ...], ...]}  (multiple episodes)
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, ep_traces in traces.items():
        color = PALETTE.get(name, "#333333")
        valid = [t for t in ep_traces if len(t) > 0]
        if not valid:
            continue
        max_len = max(len(t) for t in valid)
        # Pad shorter episodes with their last value so all traces are the same length
        padded = np.array([
            list(t) + [t[-1]] * (max_len - len(t)) for t in valid
        ], dtype=np.float64)
        arr = padded

        # Median + IQR
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        days = np.arange(len(med))

        log_med = np.log10(np.maximum(med, 1.0))
        log_q25 = np.log10(np.maximum(q25, 1.0))
        log_q75 = np.log10(np.maximum(q75, 1.0))

        ax.plot(days, log_med, color=color, linewidth=2,
                label=_pretty_name(name))
        ax.fill_between(days, log_q25, log_q75, color=color, alpha=0.12)

    ax.axhline(np.log10(target_load), color="#E24B4A", linestyle="--",
               linewidth=1.2, label="Clearance threshold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Bacterial load (log₁₀ CFU/mL)")
    ax.set_title("Bacterial load trajectories (median ± IQR)")
    ax.legend()

    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ── Figure 5: Resistance heatmap ──────────────────────────────────────────────

def plot_resistance_heatmap(
    dose_grid: np.ndarray,
    resistance_grid: np.ndarray,
    output_path: str = "figures/fig5_resistance_heatmap.pdf",
) -> plt.Figure:
    """
    Parameters
    ----------
    dose_grid       : shape (n_days, n_dose_levels) — dose administered
    resistance_grid : shape (n_days, n_dose_levels) — mean resistance level
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(
        resistance_grid.T, aspect="auto", origin="lower",
        cmap="YlOrRd", vmin=0, vmax=4,
        extent=[0, resistance_grid.shape[0], 0, resistance_grid.shape[1]]
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean resistance level (0–4)")

    dose_labels = ["0.0", "0.5", "1.0", "1.5", "2.0"]
    ax.set_yticks(np.arange(len(dose_labels)) + 0.5)
    ax.set_yticklabels(dose_labels)
    ax.set_xlabel("Treatment day")
    ax.set_ylabel("Dose (mg/kg)")
    ax.set_title("Resistance emergence by dose and day")

    fig.tight_layout()
    _save(fig, output_path)
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pretty_name(key: str) -> str:
    return {
        "adversarial_ppo": "Adversarial PPO (ours)",
        "fixed_ppo":       "Fixed-resistance PPO",
        "cycling":         "Cycling heuristic",
        "bandit":          "Contextual bandit",
        "max_dose":        "Max dose",
        "no_dose":         "No treatment",
    }.get(key, key.replace("_", " ").title())


def _save(fig: plt.Figure, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    log.info(f"Saved figure: {p}")
    plt.close(fig)
