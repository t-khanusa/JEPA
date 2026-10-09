from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple, Union


VALID_NORMS: Tuple[str, ...] = ("d_model", "v_geo", "d_t")
VALID_PROGRESS_MODES: Tuple[str, ...] = ("endpoint", "schedule_asym")
# Transverse normalization choices for LyapunovReachabilityLoss:
#   "L"   : classic scale-invariant form V_\perp = ||e_t||^2 / L^2
#   "d_t" : angle form V_\perp = ||e_t||^2 / ||d_t||^2 = sin^2 \theta_t, bounded
#           in [0, 1] (better-conditioned gradients), with anchor positions
#           (t = u_s, a_s where d_t = 0) masked out of the transverse term.
VALID_REACH_TRANSVERSE_NORMS: Tuple[str, ...] = ("L", "d_t")


class LyapunovControlLoss(nn.Module):
    """Discrete-time Lyapunov tube loss for token trajectories.

    Parameters
    ----------
    gamma : float, default 0.9
        Contraction factor in ``V_{t+1} <= gamma * V_t + tau``.
    tau : float, default 1e-4
        Additive slack / disturbance bound.
    norm_mode : {"d_model", "v_geo", "d_t"}, default "d_t"
        Choice of denominator ``N_t`` for ``V_t = ||e_t||^2 / N_t``.
    use_softplus : bool, default True
        If True use ``F.softplus(.)`` surrogate (smooth, always-on gradient);
        if False use ``F.relu(.)``.
    eps : float, default 1e-8
        Numerical floor for denominators and chord normalization.
    anchor_eps : float, default 1e-3
        Anchor-smoothing coefficient for ``norm_mode="d_t"``. Replaces the old
        hard ``max(||d_t||^2, eps=1e-8)`` clamp with a *scale-invariant* soft
        gate: ``V_t = ||e_t||^2 / (anchor_eps * L^2 + ||d_t||^2)`` where
        ``L = ||v_geo||``. Motivation: the angle form ``V_t = sin^2(theta_t)``
        is well-behaved at the exact anchors (``d_t = 0 => e_t = 0`` too, since
        ``e_t`` is the orthogonal residual of ``d_t``), but tokens a *short
        hop* from the anchor can have ``||d_t||^2`` well below the 1e-8 floor
        while still carrying a macroscopic residual ``||e_t||^2``. In that
        regime the legacy clamp produced ``V_t ~ ||e_t||^2 / 1e-8``, i.e.
        violation magnitudes that scaled with the floor rather than with the
        trajectory, dominating the loss and destroying per-token gradient
        balance. The soft gate interpolates continuously from
        ``V_t = sin^2(theta_t)`` (when ``||d_t||^2 >> anchor_eps * L^2``,
        recovered for tokens far from the anchor) to a Tikhonov
        ``v_geo`` fallback ``V_t ~ ||e_t||^2 / (anchor_eps * L^2)`` near the
        anchor. Because ``anchor_eps * L^2`` rescales with ``h``, the full
        loss inherits the same scale invariance as the pure angle form.
        Setting ``anchor_eps=0`` recovers the legacy hard-clamp behaviour.
        Has no effect when ``norm_mode != "d_t"``.
    """

    def __init__(
        self,
        gamma: float = 0.9,
        tau: float = 1e-4,
        norm_mode: str = "d_t",
        use_softplus: bool = True,
        eps: float = 1e-8,
        anchor_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if norm_mode not in VALID_NORMS:
            raise ValueError(
                f"norm_mode must be one of {VALID_NORMS}; got {norm_mode!r}"
            )
        if float(anchor_eps) < 0:
            raise ValueError(
                f"anchor_eps must be non-negative; got {anchor_eps}"
            )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.norm_mode = norm_mode
        self.use_softplus = bool(use_softplus)
        self.eps = float(eps)
        self.anchor_eps = float(anchor_eps)

    @staticmethod
    def _as_int_bounds(
        bounds: Sequence[Sequence[int]],
        name: str,
        B: int,
    ) -> List[Tuple[int, int]]:
        """Coerce a list/tensor/list-of-lists of (start, end) to Python ints."""
        if len(bounds) != B:
            raise ValueError(
                f"{name} must have length B={B}; got {len(bounds)}"
            )
        out: List[Tuple[int, int]] = []
        for i, pair in enumerate(bounds):
            if torch.is_tensor(pair):
                if pair.numel() != 2:
                    raise ValueError(
                        f"{name}[{i}] must have 2 elements; got {pair.numel()}"
                    )
                s = int(pair[0].item())
                e = int(pair[1].item())
            else:
                if len(pair) != 2:
                    raise ValueError(
                        f"{name}[{i}] must have 2 elements; got {len(pair)}"
                    )
                s = int(pair[0])
                e = int(pair[1])
            out.append((s, e))
        return out

    def forward(
        self,
        h: torch.Tensor,
        user_bounds: Sequence[Sequence[int]],
        assistant_bounds: Sequence[Sequence[int]],
        *,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the Lyapunov tube surrogate loss.

        Parameters
        ----------
        h : Tensor of shape (B, T, D)
            Full-sequence hidden states from the chosen decoder layer.
        user_bounds : list of (u_s, u_e)
            Inclusive token indices of the user span in ``h``; length B.
        assistant_bounds : list of (a_s, a_e)
            Inclusive token indices of the assistant span in ``h``; length B.
        return_diagnostics : bool
            If True, also return a dict of detached scalars for logging.

        Returns
        -------
        loss : scalar Tensor
            Mean softplus/relu violation over valid transitions.
        diag : dict, optional
            ``mean_V``, ``mean_V_curr``, ``mean_V_prev``,
            ``mean_raw_violation``, ``viol_rate``, ``surrogate_mean``,
            ``num_valid``, ``norm_mode``.
        """
        if h.dim() != 3:
            raise ValueError(f"h must be (B, T, D); got {tuple(h.shape)}")
        B, T, D = h.shape
        device = h.device
        dtype = h.dtype

        ub = self._as_int_bounds(user_bounds, "user_bounds", B)
        ab = self._as_int_bounds(assistant_bounds, "assistant_bounds", B)

        d_all = torch.zeros_like(h)
        v_geodesic = torch.zeros(B, D, device=device, dtype=dtype)
        valid_transitions = torch.zeros(B, T - 1, device=device, dtype=torch.bool)

        for i in range(B):
            u_s, u_e = ub[i]
            a_s, a_e = ab[i]
            if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
                # Skip malformed example; leaves d_all[i]=0 so its V is 0.
                continue
            v_user_i = h[i, u_e] - h[i, u_s]
            d_all[i, u_s : u_e + 1] = h[i, u_s : u_e + 1] - h[i, u_s]
            d_all[i, a_s : a_e + 1] = v_user_i + (h[i, a_s : a_e + 1] - h[i, a_s])
            v_geodesic[i] = v_user_i + (h[i, a_e] - h[i, a_s])
            if u_e > u_s:
                valid_transitions[i, u_s:u_e] = True
            if a_e > a_s:
                valid_transitions[i, a_s:a_e] = True

        v_dir = F.normalize(v_geodesic, p=2, dim=1, eps=self.eps)
        proj_scalar = (d_all * v_dir.unsqueeze(1)).sum(dim=-1, keepdim=True)
        p_all = proj_scalar * v_dir.unsqueeze(1)
        e_all = d_all - p_all

        e_sq = (e_all ** 2).sum(dim=-1)
        # L^2 = ||v_geo||^2 is reused by both the v_geo and d_t branches; the
        # eps floor protects against degenerate all-zero trajectories (e.g.
        # an empty span passed in via the random-span remapping, where the
        # caller collapses assistant_bounds to (a_s, a_s) to disable the
        # assistant chord and v_geo reduces to the user chord alone).
        L_sq = (v_geodesic ** 2).sum(dim=-1).clamp(min=self.eps)  # (B,)

        if self.norm_mode == "d_model":
            V_all = e_sq / float(max(D, 1))
        elif self.norm_mode == "v_geo":
            V_all = e_sq / L_sq.unsqueeze(1)
        elif self.norm_mode == "d_t":
            d_sq = (d_all ** 2).sum(dim=-1)  # (B, T)
            if self.anchor_eps > 0.0:
                # Scale-invariant soft anchor: far from the anchor (||d_t||^2
                # >> anchor_eps * L^2) this is sin^2(theta_t); close to it
                # V_t ~ ||e_t||^2 / (anchor_eps * L^2), a continuous Tikhonov
                # fallback to the v_geo norm mode instead of the legacy
                # (||d_t||^2).clamp(min=1e-8) cliff that blew the loss up to
                # O(1e8) at every anchor token.
                denom = self.anchor_eps * L_sq.unsqueeze(1) + d_sq
            else:
                # anchor_eps == 0 recovers the legacy hard-clamp form, kept
                # so existing checkpoints/ablations are still reproducible.
                denom = d_sq.clamp(min=self.eps)
            V_all = e_sq / denom
        else:
            raise AssertionError("unreachable")

        V_prev = V_all[:, :-1]
        V_curr = V_all[:, 1:]
        violation = V_curr - self.gamma * V_prev - self.tau
        surrogate = F.softplus(violation) if self.use_softplus else F.relu(violation)

        vf = valid_transitions.to(dtype)
        num_valid = valid_transitions.sum()
        if int(num_valid.item()) == 0:
            loss = h.sum() * 0.0
            if return_diagnostics:
                zero = torch.zeros((), device=device, dtype=dtype)
                return loss, {
                    "mean_V": zero,
                    "mean_V_curr": zero,
                    "mean_V_prev": zero,
                    "mean_raw_violation": zero,
                    "viol_rate": zero,
                    "surrogate_mean": zero,
                    "num_valid": torch.zeros((), device=device, dtype=torch.long),
                    "norm_mode": self.norm_mode,
                }
            return loss

        num_f = num_valid.to(dtype)
        loss = (surrogate * vf).sum() / num_f

        if not return_diagnostics:
            return loss

        mean_V_curr = (V_curr * vf).sum() / num_f
        mean_V_prev = (V_prev * vf).sum() / num_f
        mean_V = 0.5 * (mean_V_curr + mean_V_prev)
        mean_raw_violation = (violation * vf).sum() / num_f
        viol_rate = ((violation > 0).to(dtype) * vf).sum() / num_f
        return loss, {
            "mean_V": mean_V.detach(),
            "mean_V_curr": mean_V_curr.detach(),
            "mean_V_prev": mean_V_prev.detach(),
            "mean_raw_violation": mean_raw_violation.detach(),
            "viol_rate": viol_rate.detach(),
            "surrogate_mean": loss.detach(),
            "num_valid": num_valid.detach(),
            "norm_mode": self.norm_mode,
        }


class LyapunovReachabilityLoss(nn.Module):
    """Joint transverse-longitudinal Lyapunov loss (reach_JEPA).

    Unlike
    :class:`LyapunovControlLoss`, which constrains only the transverse residual
    ``e_t``, this loss defines a composite Lyapunov candidate that jointly
    controls:

    * the **transverse** coordinate (off-chord deviation)
      ``alpha * ||e_t||^2 / L^2``
    * the **longitudinal** coordinate (progress toward the answer anchor
      along the chord), shaped by ``progress_mode``.

    Progress modes
    --------------
    ``progress_mode="endpoint"`` (legacy, deprecated):
        ``V_prog_t = (1 - p_t/L)^2``. Empirically teleportation-prone: because
        the minimum is at ``p_t/L = 1`` for *every* token, the optimizer finds
        a degenerate solution where all tokens collapse toward the answer
        anchor (bimodal ``p_t/L`` distribution with mass near 1). Kept for
        ablations / backward compatibility only.

    ``progress_mode="schedule_asym"`` (default, recommended):
        ``V_prog_t = max(0, tau_t - p_t/L)^2`` where ``tau_t`` is a purely
        *structural* (token-rank) schedule
        ``tau_t = rank(t) / max(1, N - 1)``
        with ``rank`` enumerating the valid positions in user+assistant spans
        (N = (u_e - u_s + 1) + (a_e - a_s + 1)). Properties:

          (i) *Structural*: ``tau_t`` is a function of token indices only, so
              the optimizer cannot minimize the loss by reshaping ``v_geo``
              (unlike arc-length schedules).
          (ii) *Teleportation-proof*: the gradient w.r.t. ``p_t`` is **zero**
              whenever ``p_t/L >= tau_t``. There is no force pushing tokens
              toward the endpoint; only lagging tokens are pulled up, and
              only up to the schedule.
          (iii) *Linguistically sound*: accommodates non-uniform semantic
              density. Content tokens (high density) can naturally consume
              more progress than function tokens (low density); overshooting
              is free.
          (iv) *Telescoping closure still holds*:
              ``V_{t+1} <= gamma V_t + tau`` implies, for any sub-span
              ``[i, j]``, ``V_j <= gamma^{j-i} V_i + tau / (1 - gamma)``.
              Uniformly-small V then gives "local linearity everywhere"
              (STP's empirical random-span property) *as a theorem*.

    Both terms are **scale-invariant** in the hidden states (doubling every
    ``h[t]`` doubles ``d_t``, ``v_geo``, ``p_t``, and ``e_t`` identically, so
    ratios to ``L`` are unchanged). This rules out the ``d_model``-style
    clustering failure mode by construction.

    The discrete-time contraction ``V_{t+1} <= gamma * V_t + tau`` is enforced
    via the stable linear-softplus surrogate over valid transitions
    (user-internal + assistant-internal; chord gap skipped).

    Ablation-ready degenerations:
      * ``beta == 0``  : reduces to pure tube.
      * ``alpha == 0`` : pure longitudinal / progress loss (negative control).
      * ``progress_mode="endpoint"`` : reproduces the pre-fix loss (bad).

    Parameters
    ----------
    alpha : float, default 1.0
        Weight on the transverse term.
    beta : float, default 1.0
        Weight on the progress term (form set by ``progress_mode``).
    gamma : float, default 0.9
        Contraction factor in ``V_{t+1} <= gamma * V_t + tau``.
    tau : float, default 1e-4
        Additive slack.
    progress_mode : {"endpoint", "schedule_asym"}, default "schedule_asym"
        Shape of the longitudinal Lyapunov component.
    transverse_norm : {"L", "d_t"}, default "L"
        Denominator for the transverse term.
          * ``"L"``   : ``||e_t||^2 / L^2`` (classical, unbounded).
          * ``"d_t"`` : ``||e_t||^2 / ||d_t||^2 = sin^2 \theta_t`` (angle form;
            bounded in [0, 1], rotation-invariant; positions where
            ``d_t = 0`` — i.e. the span anchors ``t = u_s, a_s`` — are
            masked out of both the transverse term *and* the transition
            loss so the ratio never hits the clamp floor). Used by
            ``reach_JEPA_angle``.
    use_softplus : bool, default True
        If True, use ``F.softplus`` surrogate; else ``F.relu``.
    eps : float, default 1e-8
        Numerical floor for ``L`` and normalization.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.9,
        tau: float = 1e-4,
        progress_mode: str = "schedule_asym",
        transverse_norm: str = "L",
        use_softplus: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError(
                f"alpha and beta must be non-negative; got alpha={alpha}, beta={beta}"
            )
        if alpha == 0 and beta == 0:
            raise ValueError("at least one of alpha, beta must be positive")
        if progress_mode not in VALID_PROGRESS_MODES:
            raise ValueError(
                f"progress_mode must be one of {VALID_PROGRESS_MODES}; "
                f"got {progress_mode!r}"
            )
        if transverse_norm not in VALID_REACH_TRANSVERSE_NORMS:
            raise ValueError(
                f"transverse_norm must be one of {VALID_REACH_TRANSVERSE_NORMS}; "
                f"got {transverse_norm!r}"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.progress_mode = progress_mode
        self.transverse_norm = transverse_norm
        self.use_softplus = False#bool(use_softplus)
        self.eps = float(eps)

    _as_int_bounds = staticmethod(LyapunovControlLoss._as_int_bounds)

    def forward(
        self,
        h: torch.Tensor,
        user_bounds: Sequence[Sequence[int]],
        assistant_bounds: Sequence[Sequence[int]],
        *,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """See class docstring."""
        if h.dim() != 3:
            raise ValueError(f"h must be (B, T, D); got {tuple(h.shape)}")
        B, T, D = h.shape
        device = h.device
        dtype = h.dtype

        ub = self._as_int_bounds(user_bounds, "user_bounds", B)
        ab = self._as_int_bounds(assistant_bounds, "assistant_bounds", B)

        d_all = torch.zeros_like(h)
        v_geodesic = torch.zeros(B, D, device=device, dtype=dtype)
        valid_transitions = torch.zeros(B, T - 1, device=device, dtype=torch.bool)
        # `tau_sched[i, t]` is the structural progress target tau_t for the
        # schedule_asym mode; zero outside valid positions. Built once per
        # batch from integer token indices only (exogenous to h).
        tau_sched = torch.zeros(B, T, device=device, dtype=dtype)
        valid_positions = torch.zeros(B, T, device=device, dtype=torch.bool)

        for i in range(B):
            u_s, u_e = ub[i]
            a_s, a_e = ab[i]
            if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
                continue
            v_user_i = h[i, u_e] - h[i, u_s]
            d_all[i, u_s : u_e + 1] = h[i, u_s : u_e + 1] - h[i, u_s]
            d_all[i, a_s : a_e + 1] = v_user_i + (h[i, a_s : a_e + 1] - h[i, a_s])
            v_geodesic[i] = v_user_i + (h[i, a_e] - h[i, a_s])
            if u_e > u_s:
                valid_transitions[i, u_s:u_e] = True
            if a_e > a_s:
                valid_transitions[i, a_s:a_e] = True

            # Build the structural token-rank schedule tau_t for this sample.
            n_u = u_e - u_s + 1
            n_a = a_e - a_s + 1
            N = n_u + n_a
            if N > 1:
                denom = float(N - 1)
                tau_sched[i, u_s : u_e + 1] = torch.arange(
                    0, n_u, device=device, dtype=dtype
                ) / denom
                tau_sched[i, a_s : a_e + 1] = torch.arange(
                    n_u, N, device=device, dtype=dtype
                ) / denom
            valid_positions[i, u_s : u_e + 1] = True
            valid_positions[i, a_s : a_e + 1] = True

        L_sq = (v_geodesic * v_geodesic).sum(dim=-1).clamp(min=self.eps)
        L = L_sq.sqrt()
        v_dir = v_geodesic / L.unsqueeze(-1).clamp(min=self.eps)

        # Scalar projection p_t = <d_t, v_hat>  (B, T)
        p_all = (d_all * v_dir.unsqueeze(1)).sum(dim=-1)
        p_vec = p_all.unsqueeze(-1) * v_dir.unsqueeze(1)
        e_all = d_all - p_vec

        e_sq = (e_all ** 2).sum(dim=-1)  # (B, T)
        d_sq = (d_all * d_all).sum(dim=-1)  # (B, T); zero at span anchors

        # Scale-invariant composite Lyapunov candidate.
        #   V_t = alpha * V_\perp(t) + beta * V_\| (t)
        # V_\perp(t):
        #   transverse_norm="L"   -> ||e_t||^2 / L^2
        #   transverse_norm="d_t" -> ||e_t||^2 / ||d_t||^2 = sin^2 theta_t
        #     (bounded in [0,1]; anchor positions where d_t = 0 are masked
        #      out of both the transverse term AND the downstream transition
        #      so the clamp floor never leaks into the gradient).
        if self.transverse_norm == "L":
            transversal = e_sq / L_sq.unsqueeze(1)
            anchor_mask = torch.ones_like(d_sq, dtype=torch.bool)
        elif self.transverse_norm == "d_t":
            # A position is a "valid anchor neighborhood" if d_t is non-trivial.
            # We use a relative threshold (d_sq > eps * L_sq) so the mask is
            # scale-invariant.
            anchor_mask = d_sq > (self.eps * L_sq.unsqueeze(1))
            transversal = e_sq / d_sq.clamp(min=self.eps)
            # Zero out the anchor positions explicitly so the surrogate
            # mean (computed before masking below) doesn't pick up clamp-
            # floor noise.
            transversal = torch.where(
                anchor_mask, transversal, torch.zeros_like(transversal)
            )
        else:
            raise AssertionError("unreachable: transverse_norm was validated")
        p_over_L = p_all / L.unsqueeze(1)

        if self.progress_mode == "endpoint":
            # Legacy (1 - p_t/L)^2: pressures every token toward the answer
            # anchor. Known to be teleportation-prone.
            progress_raw = 1.0 - p_over_L
            progress_loss = progress_raw ** 2
        elif self.progress_mode == "schedule_asym":
            # max(0, tau_t - p_t/L)^2: lag-only penalty vs. token-rank
            # schedule. No gradient pressure past the schedule -> no
            # teleportation incentive. `progress_raw` is the signed gap
            # (positive = lagging) used for diagnostics.
            progress_raw = tau_sched - p_over_L
            progress_loss = progress_raw.clamp(min=0.0) ** 2
        else:
            raise AssertionError("unreachable: progress_mode was validated")

        V_all = self.alpha * transversal + self.beta * progress_loss

        V_prev = V_all[:, :-1]
        V_curr = V_all[:, 1:]
        violation = V_curr - self.gamma * V_prev - self.tau
        surrogate = F.softplus(violation) if self.use_softplus else F.relu(violation)

        # For transverse_norm="d_t", transitions starting at an anchor (V_prev
        # artificially = 0) are dropped; the contraction constraint
        # V_curr <= gamma * 0 + tau would otherwise be unreasonably tight.
        effective_transitions = valid_transitions
        if self.transverse_norm == "d_t":
            effective_transitions = effective_transitions & anchor_mask[:, :-1]
        vf = effective_transitions.to(dtype)
        num_valid = effective_transitions.sum()
        if int(num_valid.item()) == 0:
            loss = h.sum() * 0.0
            if return_diagnostics:
                zero = torch.zeros((), device=device, dtype=dtype)
                return loss, {
                    "mean_V": zero,
                    "mean_V_transversal": zero,
                    "mean_V_progress": zero,
                    "mean_p_over_L": zero,
                    "mean_e2_over_L2": zero,
                    "mean_tau": zero,
                    "mean_progress_gap": zero,
                    "lag_rate": zero,
                    "mean_raw_violation": zero,
                    "viol_rate": zero,
                    "surrogate_mean": zero,
                    "num_valid": torch.zeros((), device=device, dtype=torch.long),
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "progress_mode": self.progress_mode,
                    "transverse_norm": self.transverse_norm,
                }
            return loss

        num_f = num_valid.to(dtype)
        loss = (surrogate * vf).sum() / num_f

        if not return_diagnostics:
            return loss

        vf_full = torch.zeros_like(V_all, dtype=dtype)
        vf_full[:, :-1] = vf
        num_full = vf_full.sum().clamp(min=1.0)
        # Diagnostic mask over *positions* (including the last valid position
        # which has no outgoing transition): uses valid_positions so that
        # mean_tau and mean_p/L reflect the full trajectory, not just
        # transitions.
        vpos = valid_positions.to(dtype)
        num_pos = vpos.sum().clamp(min=1.0)

        mean_V = (V_all * vf_full).sum() / num_full
        mean_V_transversal = (self.alpha * transversal * vf_full).sum() / num_full
        mean_V_progress = (
            self.beta * progress_loss * vf_full
        ).sum() / num_full
        mean_p_over_L = (p_over_L * vpos).sum() / num_pos
        mean_e2_over_L2 = (transversal * vpos).sum() / num_pos
        mean_tau = (tau_sched * vpos).sum() / num_pos
        # Signed lag gap (positive = lagging behind schedule, negative = ahead
        # of schedule). For schedule_asym this is the pre-clamp term; for
        # endpoint mode we surface (1 - p/L) with the same sign convention.
        if self.progress_mode == "schedule_asym":
            signed_gap = progress_raw
        else:
            signed_gap = progress_raw
        mean_progress_gap = (signed_gap * vpos).sum() / num_pos
        lag_rate = ((signed_gap > 0).to(dtype) * vpos).sum() / num_pos
        mean_raw_violation = (violation * vf).sum() / num_f
        viol_rate = ((violation > 0).to(dtype) * vf).sum() / num_f
        return loss, {
            "mean_V": mean_V.detach(),
            "mean_V_transversal": mean_V_transversal.detach(),
            "mean_V_progress": mean_V_progress.detach(),
            "mean_p_over_L": mean_p_over_L.detach(),
            "mean_e2_over_L2": mean_e2_over_L2.detach(),
            "mean_tau": mean_tau.detach(),
            "mean_progress_gap": mean_progress_gap.detach(),
            "lag_rate": lag_rate.detach(),
            "mean_raw_violation": mean_raw_violation.detach(),
            "viol_rate": viol_rate.detach(),
            "surrogate_mean": loss.detach(),
            "num_valid": num_valid.detach(),
            "alpha": self.alpha,
            "beta": self.beta,
            "progress_mode": self.progress_mode,
            "transverse_norm": self.transverse_norm,
        }