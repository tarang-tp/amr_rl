"""
resistance/model/resistance_model.py

Adversarial resistance model. Serves two roles:

1. Genotype-to-phenotype predictor:
   Given a genotype feature vector (from PATRIC/CARD pretraining),
   predict the MIC and fitness cost of a resistance profile.

2. Adversarial mutation policy:
   Given the current observation (what the treatment policy sees),
   choose the most damaging resistance transition — i.e., act as the
   pathogen's mutation "strategy" to maximise resistance emergence.

In the minimax co-training loop:
  - Treatment policy maximises reward (clearance)
  - Resistance model maximises resistance emergence (minimises policy reward)

Architecture: MLP with a shared encoder + two heads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional


class ResistanceMLP(nn.Module):
    """
    Shared-encoder MLP.

    Inputs:  [treatment_obs (11), resistance_level (1), dose (1), in_msw (1)]  = 14 dims
    Outputs:
      - transition_logits: [stay, +1, +2]  (resistance state transitions)
      - mic_scale: scalar multiplier on EC50 (fitness/MIC prediction head)
    """

    def __init__(
        self,
        obs_dim: int = 14,
        hidden_dims: list[int] = None,
        n_transitions: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128]

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        # Adversarial head: transition distribution
        self.transition_head = nn.Linear(in_dim, n_transitions)

        # Phenotype head: predict MIC scale (for pretraining on PATRIC/CARD)
        self.mic_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),   # MIC scale must be positive
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        transition_logits = self.transition_head(h)
        mic_scale = self.mic_head(h).squeeze(-1)
        return transition_logits, mic_scale

    def transition_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class AdversarialResistanceModel:
    """
    Wrapper around ResistanceMLP with:
    - Adversarial update (maximise policy loss / resistance emergence)
    - Callable interface compatible with AMREnv.resistance_model
    - Checkpoint save/load
    """

    TRANSITIONS = [0, 1, 2]  # delta resistance levels

    def __init__(
        self,
        obs_dim: int = 14,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.15,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.net = ResistanceMLP(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims or [256, 256, 128],
            dropout=dropout,
        ).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self._episode_log: list[dict] = []

    def __call__(
        self,
        obs: np.ndarray,
        resistance_level: float,
        dose: float,
        in_msw: bool,
        rng: np.random.Generator,
    ) -> float:
        """
        Sample a resistance transition given the current state.
        Used as AMREnv.resistance_model.
        """
        x = self._build_input(obs, resistance_level, dose, in_msw)
        self.net.eval()
        with torch.no_grad():
            probs = self.net.transition_probs(x).cpu().numpy().squeeze()

        # Sample transition (0=stay, 1=+1, 2=+2)
        delta = rng.choice(self.TRANSITIONS, p=probs)
        new_level = float(np.clip(resistance_level + delta, 0.0, 4.0))

        # Log for adversarial update
        self._episode_log.append({
            "obs": obs.copy(),
            "resistance_level": resistance_level,
            "dose": dose,
            "in_msw": in_msw,
            "transition": delta,
            "probs": probs.copy(),
        })
        return new_level

    def adversarial_update(
        self,
        policy_losses: list[float],
        episode_logs: Optional[list[list[dict]]] = None,
    ) -> dict:
        """
        Update resistance model to maximise negative policy reward
        (i.e., maximise resistance emergence).

        Uses REINFORCE with policy_loss as the adversarial reward signal.
        Higher policy loss = worse outcome for the treatment policy = better for pathogen.
        """
        logs = episode_logs or [self._episode_log]
        if not logs or not any(logs):
            return {"adversarial_loss": 0.0}

        all_losses = []
        for ep_log, ep_pol_loss in zip(logs, policy_losses):
            if not ep_log:
                continue
            # Adversarial reward: positive when policy does poorly
            adv_reward = float(ep_pol_loss)

            for step in ep_log:
                x = self._build_input(
                    step["obs"], step["resistance_level"],
                    step["dose"], step["in_msw"]
                )
                logits, _ = self.net(x)
                log_prob = F.log_softmax(logits, dim=-1)
                action_lp = log_prob[0, step["transition"]]
                # REINFORCE: maximise E[adv_reward * log pi]
                loss = -adv_reward * action_lp
                all_losses.append(loss)

        if not all_losses:
            return {"adversarial_loss": 0.0}

        total_loss = torch.stack(all_losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self._episode_log = []
        return {"adversarial_loss": float(total_loss.item())}

    def pretrain_step(
        self,
        genotype_features: torch.Tensor,
        mic_labels: torch.Tensor,
    ) -> float:
        """
        Supervised pretraining on PATRIC/CARD genotype→MIC data.
        Only trains the MIC head.
        """
        self.net.train()
        x = genotype_features.to(self.device)
        y = mic_labels.to(self.device).float()

        _, mic_pred = self.net(x)
        loss = F.mse_loss(torch.log1p(mic_pred), torch.log1p(y))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

    def _build_input(
        self,
        obs: np.ndarray,
        resistance_level: float,
        dose: float,
        in_msw: bool,
    ) -> torch.Tensor:
        x = np.concatenate([
            obs.flatten(),
            [resistance_level / 4.0, dose / 2.0, float(in_msw)]
        ]).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)


class EC50Predictor:
    """
    Wraps a pretrained ResistanceMLP MIC head to predict EC50 multipliers
    from binary genotype feature vectors.

    Loaded from a standalone checkpoint saved by pretrain_resistance.py.
    The multiplier is used by AMREnv to scale the base EC50 at episode start,
    introducing pharmacodynamic heterogeneity across episodes.

    Usage
    -----
    predictor = EC50Predictor.load_from_checkpoint("checkpoints/ec50_predictor.pt")
    multiplier = predictor.predict_ec50_multiplier(gene_features)   # float >= 0
    """

    # Clamp multiplier to physiologically plausible range
    _MULT_MIN = 0.5
    _MULT_MAX = 20.0

    def __init__(self, net: ResistanceMLP, device: str = "cpu"):
        self.net    = net.to(torch.device(device))
        self.device = torch.device(device)

    def __call__(self, genotype_features: np.ndarray) -> float:
        return self.predict_ec50_multiplier(genotype_features)

    def predict_ec50_multiplier(self, genotype_features: np.ndarray) -> float:
        """
        Parameters
        ----------
        genotype_features : binary gene presence/absence vector, shape (n_genes,)

        Returns
        -------
        EC50 multiplier >= 0 (1.0 = no change from baseline)
        """
        x = torch.from_numpy(
            genotype_features.astype(np.float32)
        ).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            _, mic_scale = self.net(x)
        raw = float(mic_scale.item())
        return float(np.clip(raw, self._MULT_MIN, self._MULT_MAX))

    def save(self, path: str, obs_dim: int) -> None:
        """Save standalone checkpoint (weights + architecture metadata)."""
        # Infer hidden_dims from the encoder's Linear layers
        hidden_dims = [
            m.out_features
            for m in self.net.encoder.modules()
            if isinstance(m, torch.nn.Linear)
        ]
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "obs_dim":        obs_dim,
            "hidden_dims":    hidden_dims,
        }, path)

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "EC50Predictor":
        """Reconstruct from a checkpoint saved by save() or pretrain_resistance.py."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obs_dim     = ckpt["obs_dim"]
        hidden_dims = ckpt.get("hidden_dims", [256, 256, 128])
        net = ResistanceMLP(obs_dim=obs_dim, hidden_dims=hidden_dims)
        net.load_state_dict(ckpt["net_state_dict"])
        return cls(net, device)
