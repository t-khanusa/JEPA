"""Lyapunov planning objectives: ControlJEPA Reach / Heading energies on the imagined rollout.

The training-time regularizers (models/lyapunov/ts_lyapunov.py) shape the encoder on
data windows. Here the same energies are applied at planning time, on the predictor's
rollout towards the goal, so the planner is asked to produce trajectories that are
Lyapunov-stable w.r.t. the start->goal chord instead of only matching the last frame.

Rollout states z_t, t = 0..T-1 (z_0 encoded, the rest predicted), goal z_g. With h = phi(z)
(phi = encoder.agg for space="agg", flatten for space="patch") and chord c = h_g - h_0:

    reach:   V_t = alpha ||e_t||^2 / L^2 + beta [t/(T-1) - p_t/L]_+^2      (L = ||c||)
    heading: V_t = 1 - cos(h_{t+1} - h_t, c),  V_{-1} = 0
    cost:    V_t = ||z_t - z_g||^2 / ||z_0 - z_g||^2                       (CLF on planner cost)

    lyap = mean_t act(V_{t+1} - gamma V_t - tau)
    objective = last / last_0 + weight * lyap

`last` is the plan_gd terminal MSE (visual + alpha_proprio * proprio). It is divided by its
value at z_0 (detached, per episode) so `weight` is comparable across models whose feature
scales differ by >10x. With weight = 0 this has the same minimizer as the stock objective,
but Adam is invariant to the per-episode constant only while |grad| >> its eps (1e-8), so
the action iterates can differ; `last_raw` (normalize_last=False) is the stock objective.
"""

import torch
import torch.nn.functional as F

LYAP_KINDS = ("none", "reach", "heading", "cost")


def _act(x, use_softplus):
    return F.softplus(x) if use_softplus else F.relu(x)


def _recurrence(V, gamma, tau, use_softplus, anchor_zero=False):
    """V: (B, K). Mean over transitions of act(V_{k+1} - gamma V_k - tau) -> (B,)."""
    if anchor_zero:
        V_prev = torch.cat([torch.zeros_like(V[:, :1]), V[:, :-1]], dim=1)
        viol = V - gamma * V_prev - tau
    else:
        viol = V[:, 1:] - gamma * V[:, :-1] - tau
    return _act(viol, use_softplus).mean(dim=1)


def reach_energy(h, h_g, alpha=1.0, beta=0.5, eps=1e-8):
    """h: (B, T, d) rollout, h_g: (B, d) goal. Returns V: (B, T)."""
    T = h.shape[1]
    chord = h_g - h[:, 0]
    L_sq = (chord ** 2).sum(-1).clamp(min=eps)
    L = L_sq.sqrt()
    c_hat = chord / L.unsqueeze(-1)
    d = h - h[:, :1]
    p = (d * c_hat.unsqueeze(1)).sum(-1)
    e = d - p.unsqueeze(-1) * c_hat.unsqueeze(1)
    transverse = (e ** 2).sum(-1) / L_sq.unsqueeze(1)
    schedule = torch.arange(T, device=h.device, dtype=h.dtype) / (T - 1)
    lag = (schedule.unsqueeze(0) - p / L.unsqueeze(1)).clamp(min=0.0)
    return alpha * transverse + beta * lag ** 2


def heading_energy(h, h_g, eps=1e-8):
    """V_t = 1 - cos(step_t, h_g - h_0): (B, T-1)."""
    chord = h_g - h[:, 0]
    steps = h[:, 1:] - h[:, :-1]
    cos = F.cosine_similarity(steps, chord.unsqueeze(1).expand_as(steps), dim=-1, eps=eps)
    return 1.0 - cos


def create_lyapunov_objective_fn(
    wm=None,
    alpha=0,
    base=2,
    mode="last",
    lyap="none",
    space="agg",
    weight=1.0,
    gamma=0.95,
    tau=1e-4,
    lyap_alpha=1.0,
    lyap_beta=0.5,
    use_softplus=False,
    normalize_last=True,
    eps=1e-8,
):
    """Drop-in for planning.objectives.create_objective_fn (mode is only a label here)."""
    if lyap not in LYAP_KINDS:
        raise ValueError(f"lyap must be one of {LYAP_KINDS}, got {lyap!r}")
    if lyap in ("reach", "heading") and space == "agg" and not hasattr(getattr(wm, "encoder", None), "agg"):
        raise ValueError("space='agg' needs wm.encoder.agg(); pass wm or use space='patch'.")

    def embed(z_visual):
        """(B, T, P, D) -> (B, T, d) in the space the energy is measured."""
        B, T = z_visual.shape[:2]
        if space == "agg":
            return wm.encoder.agg(z_visual.reshape(B * T, *z_visual.shape[2:])).reshape(B, T, -1)
        return z_visual.reshape(B, T, -1)

    def sq_dist(a, b):
        return ((a - b) ** 2).mean(dim=tuple(range(2, a.ndim)))  # (B, T)

    def objective_fn(z_obs_pred, z_obs_tgt, step=None, return_terms=False):
        zv, zg = z_obs_pred["visual"], z_obs_tgt["visual"]  # (B, T, P, D), (B, 1, P, D)
        last = sq_dist(zv[:, -1:], zg)[:, 0] + alpha * sq_dist(
            z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]
        )[:, 0]
        cost_to_go = sq_dist(zv, zg)  # (B, T), visual only
        scale = cost_to_go[:, 0].detach().clamp(min=eps)
        loss = last / scale if normalize_last else last

        lyap_term = torch.zeros_like(loss)
        if lyap != "none" and weight > 0:
            if lyap == "cost":
                lyap_term = _recurrence(cost_to_go / scale.unsqueeze(1), gamma, tau, use_softplus)
            else:
                h = embed(zv).float()
                h_g = embed(zg).float()[:, 0]
                if lyap == "reach":
                    V = reach_energy(h, h_g, lyap_alpha, lyap_beta, eps)
                    lyap_term = _recurrence(V, gamma, tau, use_softplus)
                else:
                    V = heading_energy(h, h_g, eps)
                    lyap_term = _recurrence(V, gamma, tau, use_softplus, anchor_zero=True)
            loss = loss + weight * lyap_term.to(loss.dtype)
        if return_terms:
            return loss, {"last": last.detach(), "last_norm": (last / scale).detach(), "lyap": lyap_term.detach()}
        return loss

    return objective_fn


def parse_objective_spec(spec):
    """'last' | 'last_raw' | '<lyap>[_<space>]_w<weight>[_g<gamma>][_raw]' -> kwargs, e.g. 'heading_agg_w1'.

    'raw' turns off the per-episode normalization of the terminal cost (stock plan_gd objective).
    """
    if spec == "last":
        return {"lyap": "none", "weight": 0.0}
    if spec == "last_raw":
        return {"lyap": "none", "weight": 0.0, "normalize_last": False}
    parts = spec.split("_")
    kw = {"lyap": parts[0]}
    for p in parts[1:]:
        if p in ("agg", "patch"):
            kw["space"] = p
        elif p.startswith("w"):
            kw["weight"] = float(p[1:])
        elif p.startswith("g"):
            kw["gamma"] = float(p[1:])
        elif p == "sp":
            kw["use_softplus"] = True
        elif p == "raw":
            kw["normalize_last"] = False
        else:
            raise ValueError(f"Unknown token {p!r} in objective spec {spec!r}")
    return kw
