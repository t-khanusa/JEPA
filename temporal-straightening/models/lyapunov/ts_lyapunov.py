from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LYAPUNOV_MODES = ("lyapreach", "lyaptube", "lyapheading")


class TSLyapunovRegularizer(nn.Module):
    """ControlJEPA Lyapunov loss as a drop-in for the TS aggcos curvature term.

    Input h: (B, T, d) aggregated latents of one training window (T = num_hist + num_pred).
    The window is a single start->goal segment with chord c = h_{T-1} - h_0:

        d_t = h_t - h_0,  p_t = <d_t, c_hat>,  e_t = d_t - p_t c_hat,  L = ||c||
        V_t = alpha * ||e_t||^2 / L^2 + beta * max(0, t/(T-1) - p_t/L)^2
        loss = mean_t act(V_{t+1} - gamma * V_t - tau)

    mode="lyaptube" forces beta = 0 (pure transverse tube, LyapunovControlLoss with
    norm_mode="v_geo"). mode="lyapreach" is LyapunovReachabilityLoss with the
    progress schedule t/(T-1) and transverse_norm="L".

    mode="lyapheading" uses the heading error of every step against the chord,
        V_t = 1 - cos(h_{t+1} - h_t, c),  t = 0..T-2,  V_{-1} = 0,
    with the same recurrence. V is bounded in [0, 2], invariant to the scale of each
    step and to the chord length, and penalizes backtracking along the chord.

    Windows with L^2 < min_L_sq carry no chord direction and are excluded.
    """

    def __init__(
        self,
        mode: str = "lyapreach",
        gamma: float = 0.95,
        tau: float = 1e-4,
        alpha: float = 1.0,
        beta: float = 0.5,
        min_L_sq: float = 1e-8,
        use_softplus: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        if mode not in LYAPUNOV_MODES:
            raise ValueError(f"mode must be one of {LYAPUNOV_MODES}, got {mode!r}")
        self.mode = mode
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta) if mode == "lyapreach" else 0.0
        self.min_L_sq = float(min_L_sq)
        self.use_softplus = bool(use_softplus)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.dim() != 3 or h.shape[1] < 3:
            raise ValueError(f"h must be (B, T>=3, d), got {tuple(h.shape)}")
        h = h.float()
        if self.mode == "lyapheading":
            return self._heading(h)
        B, T, _ = h.shape

        chord = h[:, -1] - h[:, 0]
        L_sq = (chord ** 2).sum(-1)
        valid = L_sq > self.min_L_sq
        L_sq = L_sq.clamp(min=self.eps)
        L = L_sq.sqrt()
        c_hat = chord / L.unsqueeze(-1)

        d = h - h[:, :1]
        p = (d * c_hat.unsqueeze(1)).sum(-1)
        e = d - p.unsqueeze(-1) * c_hat.unsqueeze(1)
        transverse = (e ** 2).sum(-1) / L_sq.unsqueeze(1)
        p_over_L = p / L.unsqueeze(1)
        schedule = torch.arange(T, device=h.device, dtype=h.dtype) / (T - 1)
        lag = schedule.unsqueeze(0) - p_over_L
        progress = lag.clamp(min=0.0) ** 2

        V = self.alpha * transverse + self.beta * progress
        violation = V[:, 1:] - self.gamma * V[:, :-1] - self.tau
        surrogate = F.softplus(violation) if self.use_softplus else F.relu(violation)

        w = valid.to(h.dtype).unsqueeze(1).expand_as(violation)
        n = w.sum().clamp(min=1.0)
        loss = (surrogate * w).sum() / n

        with torch.no_grad():
            wi = valid.to(h.dtype).unsqueeze(1).expand(B, T - 2)
            ni = wi.sum().clamp(min=1.0)
            interior = slice(1, T - 1)
            logs = {
                "lyap_V_mean": (V[:, 1:] * w).sum() / n,
                "lyap_V_transverse": (transverse[:, interior] * wi).sum() / ni,
                "lyap_V_progress": (progress[:, interior] * wi).sum() / ni,
                "lyap_p_over_L": (p_over_L[:, interior] * wi).sum() / ni,
                "lyap_lag_rate": ((lag[:, interior] > 0).to(h.dtype) * wi).sum() / ni,
                "lyap_viol_rate": ((violation > 0).to(h.dtype) * w).sum() / n,
                "lyap_raw_violation": (violation * w).sum() / n,
                "lyap_valid_frac": valid.to(h.dtype).mean(),
                "lyap_chord_norm": L.mean(),
            }
        return loss, logs

    def _heading(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        chord = h[:, -1] - h[:, 0]
        valid = (chord ** 2).sum(-1) > self.min_L_sq
        steps = h[:, 1:] - h[:, :-1]
        cos = F.cosine_similarity(steps, chord.unsqueeze(1).expand_as(steps), dim=-1, eps=self.eps)
        V = 1.0 - cos
        V_prev = torch.cat([torch.zeros_like(V[:, :1]), V[:, :-1]], dim=1)
        violation = V - self.gamma * V_prev - self.tau
        surrogate = F.softplus(violation) if self.use_softplus else F.relu(violation)

        w = valid.to(h.dtype).unsqueeze(1).expand_as(violation)
        n = w.sum().clamp(min=1.0)
        loss = (surrogate * w).sum() / n

        with torch.no_grad():
            logs = {
                "lyap_V_mean": (V * w).sum() / n,
                "lyap_heading_cos": (cos * w).sum() / n,
                "lyap_backtrack_rate": ((cos < 0).to(h.dtype) * w).sum() / n,
                "lyap_viol_rate": ((violation > 0).to(h.dtype) * w).sum() / n,
                "lyap_raw_violation": (violation * w).sum() / n,
                "lyap_valid_frac": valid.to(h.dtype).mean(),
            }
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

    dist = (h - h[:, -1:]).norm(dim=-1)
    monotone = (dist[:, 1:] < dist[:, :-1]).float()

    chord = h[:, -1] - h[:, 0]
    L = chord.norm(dim=-1, keepdim=True).clamp(min=eps)
    c_hat = chord / L
    d = h - h[:, :1]
    p = (d * c_hat.unsqueeze(1)).sum(-1)
    e = d - p.unsqueeze(-1) * c_hat.unsqueeze(1)
    V = (e ** 2).sum(-1) / (L ** 2)
    p_over_L = p / L

    path_len = v.norm(dim=-1).sum(-1)

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
