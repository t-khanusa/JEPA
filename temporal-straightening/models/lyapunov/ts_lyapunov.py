from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .planning_losses import PlanningLyapunovReachLoss, PlanningLyapunovRolloutTubeLoss

LYAPUNOV_MODES = ("lyapreach", "lyaptube")


def _tensor_diagnostics(diag: Dict, prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in diag.items():
        if torch.is_tensor(v):
            out[f"{prefix}{k}"] = v.detach().float()
    return out


class TSLyapunovRegularizer(nn.Module):
    """Lyapunov replacement for the TS cosine-curvature term.

    Operates on aggregated per-frame latents h: (B, T, d). Each window is one
    start->goal segment with chord v_geo = h_{T-1} - h_0, matching the planning
    objective, which compares the terminal latent against the goal latent.

    mode="lyapreach": transverse tube + lag-only progress toward the window end
        (LyapunovReachabilityLoss, schedule_asym, transverse_norm configurable).
    mode="lyaptube":  pure transverse tube (LyapunovControlLoss, norm v_geo).
    rollout_weight > 0 additionally applies the tube to the predictor's
    one-step outputs, anchored at the encoded start and a stop-grad goal.
    """

    def __init__(
        self,
        mode: str = "lyapreach",
        gamma: float = 0.95,
        tau: float = 1e-4,
        alpha: float = 1.0,
        beta: float = 0.5,
        min_L_sq: float = 1e-8,
        transverse_norm: str = "L",
        rollout_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if mode not in LYAPUNOV_MODES:
            raise ValueError(f"mode must be one of {LYAPUNOV_MODES}, got {mode!r}")
        self.mode = mode
        if transverse_norm not in ("L", "d_t"):
            raise ValueError(f"transverse_norm must be 'L' or 'd_t', got {transverse_norm!r}")
        self.rollout_weight = float(rollout_weight)
        if mode == "lyapreach":
            self.loss_fn = PlanningLyapunovReachLoss(
                gamma=float(gamma),
                tau=float(tau),
                alpha=float(alpha),
                beta=float(beta),
                min_L_sq=float(min_L_sq),
            )
            self.loss_fn.loss_fn.transverse_norm = str(transverse_norm)
        else:
            self.loss_fn = PlanningLyapunovRolloutTubeLoss(
                gamma=float(gamma), tau=float(tau), min_L_sq=float(min_L_sq)
            )
        self.rollout_loss_fn = PlanningLyapunovRolloutTubeLoss(
            gamma=float(gamma), tau=float(tau), min_L_sq=float(min_L_sq)
        )

    def forward(
        self,
        h: torch.Tensor,
        h_pred: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.shape[1] < 3:
            raise ValueError(f"Lyapunov loss needs >= 3 frames, got {h.shape[1]}")
        horizon = h.shape[1] - 1
        loss, diag = self.loss_fn(h, horizon, return_diagnostics=True)
        logs = _tensor_diagnostics(diag, "lyap_")
        logs["lyap_loss_demo"] = loss.detach().float()

        if self.rollout_weight > 0 and h_pred is not None:
            r_loss, r_diag = self.rollout_loss_fn(h_pred, horizon, return_diagnostics=True)
            logs.update(_tensor_diagnostics(r_diag, "lyap_rollout_"))
            logs["lyap_loss_rollout"] = r_loss.detach().float()
            loss = loss + self.rollout_weight * r_loss
        return loss, logs


@torch.no_grad()
def trajectory_geometry_metrics(h: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Loss-agnostic geometry of aggregated latent windows h: (B, T, d).

    Logged identically for cosine and Lyapunov runs so the two regularizers
    are compared on the same axes rather than on their own training losses.
    """
    h = h.float()
    v = h[:, 1:] - h[:, :-1]
    cos = F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1, eps=eps)

    goal = h[:, -1:]
    dist = (h - goal).norm(dim=-1)
    monotone = (dist[:, 1:] < dist[:, :-1]).float()

    v_geo = h[:, -1] - h[:, 0]
    L = v_geo.norm(dim=-1, keepdim=True).clamp(min=eps)
    v_hat = v_geo / L
    d = h - h[:, :1]
    p = (d * v_hat.unsqueeze(1)).sum(-1)
    e_sq = (d ** 2).sum(-1) - p ** 2
    V = e_sq.clamp(min=0) / (L ** 2)
    p_over_L = p / L

    step = v.norm(dim=-1)
    path_len = step.sum(-1)

    return {
        "geo_cos_sim": cos.mean(),
        "geo_curvature": (1.0 - cos).mean(),
        "geo_goal_monotone_rate": monotone.mean(),
        "geo_V_transverse_mean": V[:, 1:-1].mean(),
        "geo_V_transverse_max": V[:, 1:-1].amax(dim=-1).mean(),
        "geo_p_over_L_mid": p_over_L[:, 1:-1].mean(),
        "geo_progress_monotone_rate": (p_over_L[:, 1:] > p_over_L[:, :-1]).float().mean(),
        "geo_chord_over_path": (L.squeeze(-1) / path_len.clamp(min=eps)).mean(),
        "geo_latent_std": h.reshape(-1, h.shape[-1]).std(dim=0).mean(),
    }
