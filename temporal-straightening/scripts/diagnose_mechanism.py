"""Mechanism diagnostics: aggcos vs Lyapunov reach on the same checkpoint (eval only).

A. Window geometry on training-length windows (T = num_hist + num_pred frames):
   both regularizers evaluated per window on agg latents and on flattened patches,
   tail concentration, and rank correlation with physical (state-space) geometry.
B. Gradient alignment at the checkpoint parameters: cosine between the gradients of
   z_loss (JEPA), aggcos and reach w.r.t. projector and agg-head parameters.
C. Planner-space information: linear probe of agent xy from patches / agg, and
   faithfulness of latent goal distance to physical goal distance over H model steps.

  python scripts/diagnose_mechanism.py --run checkpoints/test/<run> --epoch 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a, dim=-1), dim=-1).float()
    rb = torch.argsort(torch.argsort(b, dim=-1), dim=-1).float()
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp(min=1e-12)


def cos_curv(h, eps=1e-6):
    v = h[:, 1:] - h[:, :-1]
    return (1 - F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1, eps=eps)).mean(-1)


def reach_terms(h, gamma=0.95, tau=1e-4, alpha=1.0, beta=0.5, eps=1e-8):
    T = h.shape[1]
    c = h[:, -1] - h[:, 0]
    L_sq = (c ** 2).sum(-1).clamp(min=eps)
    c_hat = c / L_sq.sqrt().unsqueeze(-1)
    d = h - h[:, :1]
    p = (d * c_hat.unsqueeze(1)).sum(-1)
    e = d - p.unsqueeze(-1) * c_hat.unsqueeze(1)
    trans = (e ** 2).sum(-1) / L_sq.unsqueeze(1)
    sched = torch.arange(T, device=h.device, dtype=h.dtype) / (T - 1)
    prog = (sched - p / L_sq.sqrt().unsqueeze(1)).clamp(min=0) ** 2
    V = alpha * trans + beta * prog
    viol = V[:, 1:] - gamma * V[:, :-1] - tau
    return F.relu(viol).mean(-1), trans[:, 1:-1].max(-1).values, L_sq.sqrt(), (viol > 0).float().mean(-1)


def tail_share(x, q=0.1):
    k = max(1, int(len(x) * q))
    return (x.sort(descending=True).values[:k].sum() / x.sum().clamp(min=1e-12)).item()


def ridge_r2(X, Y, lam=1e-2):
    n = X.shape[0] // 2
    mu, sd = X[:n].mean(0), X[:n].std(0).clamp(min=1e-6)
    Xs = (X - mu) / sd
    Xtr, Xte, Ytr, Yte = Xs[:n], Xs[n:], Y[:n], Y[n:]
    Xtr1 = torch.cat([Xtr, torch.ones(n, 1)], 1)
    Xte1 = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1)
    A = Xtr1.T @ Xtr1 + lam * n * torch.eye(Xtr1.shape[1])
    W = torch.linalg.solve(A, Xtr1.T @ Ytr)
    res = ((Xte1 @ W - Yte) ** 2).sum(0)
    tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return (1 - res / tot).mean().item()


def find_attr(obj, name):
    while obj is not None:
        if hasattr(obj, name):
            return getattr(obj, name)
        obj = getattr(obj, "dataset", None)
    raise AttributeError(name)


def sample_windows(dset, n, n_frames, fs, rng):
    out = []
    while len(out) < n:
        obs, act, state, _ = dset[int(rng.integers(len(dset)))]
        T = obs["visual"].shape[0]
        span = (n_frames - 1) * fs
        if T <= span + fs:
            continue
        s = int(rng.integers(0, T - span - fs))
        o = {k: v[s : s + span + 1 : fs] for k, v in obs.items()}
        a = rearrange(act[s : s + n_frames * fs], "(h f) d -> h (f d)", f=fs)
        out.append((o, a, state[s : s + span + 1 : fs, :2].float()))
    return out


def stack(chunk, device):
    obs = {k: torch.stack([w[0][k] for w in chunk]).to(device) for k in chunk[0][0]}
    act = torch.stack([w[1] for w in chunk]).to(device)
    xy = torch.stack([w[2] for w in chunk])
    return obs, act, xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--n_windows", type=int, default=512)
    ap.add_argument("--n_long", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n_grad_batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import hydra
    from omegaconf import OmegaConf
    import custom_resolvers  # noqa: F401
    from plan import load_model

    run = Path(args.run).resolve()
    cfg = OmegaConf.load(run / "hydra.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, traj = hydra.utils.call(
        cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip
    )
    dset = traj["valid"]
    model = load_model(run / "checkpoints" / f"model_{args.epoch}.pth", cfg, cfg.num_action_repeat, device)
    model.eval()
    enc = model.encoder
    fs, T = cfg.frameskip, cfg.num_hist + cfg.num_pred
    rng = np.random.default_rng(args.seed)
    out = {"run": run.name, "epoch": args.epoch}

    # A. window geometry on training-length windows
    wins = sample_windows(dset, args.n_windows, T, fs, rng)
    H_agg, Z_flat, XY = [], [], []
    with torch.no_grad():
        for i in range(0, len(wins), args.batch):
            obs, act, xy = stack(wins[i : i + args.batch], device)
            zv = model.encode_obs(obs)["visual"].float()
            b, t, p, d = zv.shape
            H_agg.append(enc.agg(zv.reshape(b * t, p, d)).reshape(b, t, -1).cpu())
            Z_flat.append(zv.reshape(b, t, -1).cpu())
            XY.append(xy)
    H_agg, Z_flat, XY = torch.cat(H_agg), torch.cat(Z_flat), torch.cat(XY)

    phys_step = (XY[:, 1:] - XY[:, :-1]).norm(dim=-1)
    moving = phys_step.min(-1).values > 1e-3
    phys_curv = cos_curv(XY.double()).float()
    for space, Hs in [("agg", H_agg), ("patch", Z_flat)]:
        cc = cos_curv(Hs)
        rl, tmax, L, vr = reach_terms(Hs)
        out[f"A_{space}_cos_loss"] = cc.mean().item()
        out[f"A_{space}_reach_loss"] = rl.mean().item()
        out[f"A_{space}_reach_viol_rate"] = vr.mean().item()
        out[f"A_{space}_cos_top10_share"] = tail_share(cc)
        out[f"A_{space}_reach_top10_share"] = tail_share(rl)
        out[f"A_{space}_spearman_cos_vs_reach"] = spearman(cc[None], rl[None]).item()
        out[f"A_{space}_spearman_reach_vs_chord"] = spearman(rl[None], L[None]).item()
        out[f"A_{space}_spearman_cos_vs_physcurv"] = spearman(cc[moving][None], phys_curv[moving][None]).item()
        out[f"A_{space}_spearman_reach_vs_physcurv"] = spearman(rl[moving][None], phys_curv[moving][None]).item()
        straight = moving & (phys_curv < 0.05)
        out[f"A_{space}_cos_loss_on_phys_straight"] = cc[straight].mean().item()
        out[f"A_{space}_reach_loss_on_phys_straight"] = rl[straight].mean().item()
    out["A_frac_moving"] = moving.float().mean().item()
    out["A_frac_phys_straight"] = (moving & (phys_curv < 0.05)).float().mean().item()

    # B. gradient alignment at these parameters
    groups = {
        "proj": [p for n, p in enc.named_parameters() if "projector" in n],
        "agg": [p for n, p in enc.named_parameters() if n.startswith("agg_")],
    }
    for ps in groups.values():
        for p_ in ps:
            p_.requires_grad_(True)
    grads = {k: {g: None for g in groups} for k in ["jepa", "cos", "reach"]}
    gwins = sample_windows(dset, args.n_grad_batches * args.batch, T, fs, rng)
    for i in range(0, len(gwins), args.batch):
        obs, act, _ = stack(gwins[i : i + args.batch], device)
        z = model.encode(obs, act)
        z_pred = model.predict(z[:, : model.num_hist])
        z_tgt = z[:, model.num_pred :].detach()
        jepa = F.mse_loss(z_pred[..., : -model.action_dim], z_tgt[..., : -model.action_dim])
        zv = model.visual_only(z)
        b, t, p, d = zv.shape
        h = enc.agg(zv.reshape(b * t, p, d)).reshape(b, t, -1).float()
        losses = {"jepa": jepa, "cos": cos_curv(h).mean(), "reach": reach_terms(h)[0].mean()}
        for name, loss in losses.items():
            for g, ps in groups.items():
                gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
                flat = torch.cat([(x if x is not None else torch.zeros_like(p_)).flatten() for x, p_ in zip(gs, ps)])
                grads[name][g] = flat if grads[name][g] is None else grads[name][g] + flat
    for g in groups:
        for a_, b_ in [("cos", "reach"), ("jepa", "cos"), ("jepa", "reach")]:
            out[f"B_{g}_gradcos_{a_}_{b_}"] = F.cosine_similarity(grads[a_][g], grads[b_][g], dim=0).item()
        for k in grads:
            out[f"B_{g}_gradnorm_{k}"] = grads[k][g].norm().item() / args.n_grad_batches
        out[f"B_{g}_ratio_0.1cos_over_jepa"] = 0.1 * grads["cos"][g].norm().item() / max(grads["jepa"][g].norm().item(), 1e-12)
        out[f"B_{g}_ratio_0.1reach_over_jepa"] = 0.1 * grads["reach"][g].norm().item() / max(grads["jepa"][g].norm().item(), 1e-12)

    # C. planner-space information and metric faithfulness over H model steps
    frames_z = Z_flat.reshape(-1, Z_flat.shape[-1])
    frames_h = H_agg.reshape(-1, H_agg.shape[-1])
    frames_xy = XY.reshape(-1, 2)
    perm = torch.randperm(frames_z.shape[0], generator=torch.Generator().manual_seed(0))
    out["C_probe_r2_xy_patch"] = ridge_r2(frames_z[perm], frames_xy[perm])
    out["C_probe_r2_xy_agg"] = ridge_r2(frames_h[perm], frames_xy[perm])
    lwins = sample_windows(dset, args.n_long, args.horizon + 1, fs, rng)
    Zl, Hl, XYl = [], [], []
    with torch.no_grad():
        for i in range(0, len(lwins), args.batch):
            obs, act, xy = stack(lwins[i : i + args.batch], device)
            zv = model.encode_obs(obs)["visual"].float()
            b, t, p, d = zv.shape
            Hl.append(enc.agg(zv.reshape(b * t, p, d)).reshape(b, t, -1).cpu())
            Zl.append(zv.reshape(b, t, -1).cpu())
            XYl.append(xy)
    Zl, Hl, XYl = torch.cat(Zl), torch.cat(Hl), torch.cat(XYl)
    phys_d = (XYl[:, :-1] - XYl[:, -1:]).norm(dim=-1)
    for space, Hs in [("patch", Zl), ("agg", Hl)]:
        lat_d = (Hs[:, :-1] - Hs[:, -1:]).norm(dim=-1)
        out[f"C_{space}_spearman_goaldist_vs_phys"] = spearman(lat_d, phys_d).nanmean().item()
        steps = torch.arange(args.horizon, 0, -1).float().expand_as(lat_d)
        out[f"C_{space}_spearman_goaldist_vs_steps"] = spearman(lat_d, steps).nanmean().item()
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
