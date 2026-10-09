from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from controlJEPA import LyapunovControlLoss, LyapunovReachabilityLoss


def planning_segment_bounds(horizon: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Single start→goal span mapped onto ControlJEPA's two-span API."""
    return [(0, horizon)], [(horizon, horizon)]


def visual_to_agg_latents(encoder, visual_feats: torch.Tensor) -> torch.Tensor:
    """(B, T, P, D) patch visuals -> (B, T, d) aggregated latents."""
    if not hasattr(encoder, "agg"):
        raise ValueError("Lyapunov planning requires encoder.agg().")
    b, t, p, d = visual_feats.shape
    tokens = visual_feats.reshape(b * t, p, d)
    return encoder.agg(tokens).reshape(b, t, -1)


def filter_by_geodesic_length(
    h: torch.Tensor,
    horizon: int,
    min_L_sq: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep batch items with ||h_H - h_0||^2 > min_L_sq."""
    v_geo = h[:, horizon] - h[:, 0]
    L_sq = (v_geo ** 2).sum(dim=-1)
    valid = L_sq > min_L_sq
    if not valid.any():
        return h[:0], valid
    return h[valid], valid


class PlanningLyapunovReachLoss(nn.Module):
    """P1: Reach-JEPA on encoded demo trajectories (norm v_geo via transverse_norm='L')."""

    def __init__(
        self,
        gamma: float = 0.95,
        tau: float = 1e-4,
        alpha: float = 1.0,
        beta: float = 0.5,
        min_L_sq: float = 1e-8,
    ) -> None:
        super().__init__()
        self.min_L_sq = float(min_L_sq)
        self.loss_fn = LyapunovReachabilityLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau=tau,
            progress_mode="schedule_asym",
            transverse_norm="L",
            use_softplus=True,
        )

    def forward(
        self,
        h: torch.Tensor,
        horizon: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        h_seg = h[:, : horizon + 1]
        h_valid, _ = filter_by_geodesic_length(h_seg, horizon, self.min_L_sq)
        if h_valid.shape[0] == 0:
            zero = h.sum() * 0.0
            if return_diagnostics:
                return zero, {"num_valid_segments": torch.zeros(())}
            return zero

        b = h_valid.shape[0]
        user_bounds = [(0, horizon)] * b
        assistant_bounds = [(horizon, horizon)] * b
        return self.loss_fn(
            h_valid,
            user_bounds,
            assistant_bounds,
            return_diagnostics=return_diagnostics,
        )


class PlanningLyapunovRolloutTubeLoss(nn.Module):
    """P2: Pure v_geo tube on predicted rollout trajectories."""

    def __init__(
        self,
        gamma: float = 0.95,
        tau: float = 1e-4,
        min_L_sq: float = 1e-8,
    ) -> None:
        super().__init__()
        self.min_L_sq = float(min_L_sq)
        self.loss_fn = LyapunovControlLoss(
            gamma=gamma,
            tau=tau,
            norm_mode="v_geo",
            use_softplus=True,
        )

    def forward(
        self,
        h: torch.Tensor,
        horizon: int,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        h_seg = h[:, : horizon + 1]
        h_valid, _ = filter_by_geodesic_length(h_seg, horizon, self.min_L_sq)
        if h_valid.shape[0] == 0:
            zero = h.sum() * 0.0
            if return_diagnostics:
                return zero, {"num_valid_segments": torch.zeros(())}
            return zero

        b = h_valid.shape[0]
        user_bounds = [(0, horizon)] * b
        assistant_bounds = [(horizon, horizon)] * b
        return self.loss_fn(
            h_valid,
            user_bounds,
            assistant_bounds,
            return_diagnostics=return_diagnostics,
        )


def build_encoded_agg_trajectory(encoder, visual_feats: torch.Tensor, horizon: int) -> torch.Tensor:
    """Encoded demo latents h_0..h_H from ground-truth visuals."""
    h = visual_to_agg_latents(encoder, visual_feats)
    return h[:, : horizon + 1]


def build_predicted_agg_trajectory(
    wm,
    z: torch.Tensor,
    act: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Rollout tube latents: encoded h_0, predicted h_1..h_{H-1}, stop-grad h_H."""
    visual = wm.visual_only(z)
    h_start = visual_to_agg_latents(wm.encoder, visual[:, 0:1])

    h_steps: List[torch.Tensor] = [h_start]
    z_roll = z[:, : wm.num_hist].clone()

    for t in range(1, horizon):
        if t < wm.num_hist:
            h_t = visual_to_agg_latents(wm.encoder, visual[:, t : t + 1])
            h_steps.append(h_t)
            if z_roll.shape[1] <= t:
                z_roll = torch.cat([z_roll, z[:, t : t + 1]], dim=1)
        else:
            z_pred = wm.predict(z_roll[:, -wm.num_hist :])
            z_new = z_pred[:, -1:, ...]
            z_new = wm.replace_actions_from_z(z_new, act[:, t : t + 1])
            h_t = visual_to_agg_latents(wm.encoder, wm.visual_only(z_new))
            h_steps.append(h_t)
            z_roll = torch.cat([z_roll, z_new], dim=1)

    h_goal = visual_to_agg_latents(wm.encoder, visual[:, horizon : horizon + 1]).detach()
    h_steps.append(h_goal)
    return torch.cat(h_steps, dim=1)