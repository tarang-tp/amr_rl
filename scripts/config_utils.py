"""scripts/config_utils.py — shared YAML config loader with type coercion.

PyYAML safe_load can return numeric-looking values (e.g. 1.0e8) as strings
depending on the installed version and locale. All scripts should load config
through load_config() rather than calling yaml.safe_load directly.
"""

from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _cast(cfg)


def _cast(cfg: dict) -> dict:
    env = cfg.get("env", {})
    env["max_episode_steps"]  = int(env.get("max_episode_steps", 14))
    env["bacterial_load_init"] = float(env.get("bacterial_load_init", 1e8))
    env["target_load"]        = float(env.get("target_load", 1e3))
    env["max_dose"]           = float(env.get("max_dose", 400.0))

    pol = cfg.get("policy", {})
    pol["learning_rate"]   = float(pol.get("learning_rate", 3e-4))
    pol["n_steps"]         = int(pol.get("n_steps", 4096))
    pol["batch_size"]      = int(pol.get("batch_size", 512))
    pol["n_epochs"]        = int(pol.get("n_epochs", 10))
    pol["gamma"]           = float(pol.get("gamma", 0.99))
    pol["gae_lambda"]      = float(pol.get("gae_lambda", 0.95))
    pol["clip_range"]      = float(pol.get("clip_range", 0.2))
    pol["ent_coef"]        = float(pol.get("ent_coef", 0.01))
    pol["vf_coef"]         = float(pol.get("vf_coef", 0.5))
    pol["max_grad_norm"]   = float(pol.get("max_grad_norm", 0.5))
    pol["total_timesteps"] = int(pol.get("total_timesteps", 2_000_000))
    pol["net_arch"]        = [int(x) for x in pol.get("net_arch", [256, 256])]

    res = cfg.get("resistance", {})
    res["fitness_cost_slope"]      = float(res.get("fitness_cost_slope", 0.08))
    res["adversarial_lr"]          = float(res.get("adversarial_lr", 3e-4))
    res["adversarial_update_freq"] = int(res.get("adversarial_update_freq", 4))
    res["hidden_dims"]             = [int(x) for x in res.get("hidden_dims", [256, 256, 128])]
    res["dropout"]                 = float(res.get("dropout", 0.15))

    adv = cfg.get("adversarial", {})
    adv["co_train_ratio"] = int(adv.get("co_train_ratio", 4))

    rew = cfg.get("reward", {})
    if "w_clearance"  in rew: rew["w_clearance"]  = float(rew["w_clearance"])
    if "w_load"       in rew: rew["w_load"]        = float(rew["w_load"])
    if "w_dose"       in rew: rew["w_dose"]        = float(rew["w_dose"])
    if "w_resistance" in rew: rew["w_resistance"]  = float(rew["w_resistance"])
    if "msw_penalty"  in rew: rew["msw_penalty"]   = float(rew["msw_penalty"])
    if "w_progress"   in rew: rew["w_progress"]    = float(rew["w_progress"])

    bl = cfg.get("baselines", {})
    bl["cycling_period"]  = int(bl.get("cycling_period", 3))
    bl["bandit_epsilon"]  = float(bl.get("bandit_epsilon", 0.1))

    ev = cfg.get("eval", {})
    ev["n_eval_episodes"]  = int(ev.get("n_eval_episodes", 500))
    ev["checkpoint_freq"]  = int(ev.get("checkpoint_freq", 100_000))

    return cfg
