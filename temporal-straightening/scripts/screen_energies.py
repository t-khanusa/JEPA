"""Offline screen of candidate Lyapunov energies on a frozen checkpoint (eval only).

For each candidate regularizer on agg latents of training-length windows, reports:
rank correlation with physical curvature, top-10% loss share (tail concentration),
projector-gradient norm relative to the JEPA gradient (x lambda=0.1), and gradient
cosine with aggcos and with JEPA.

  python scripts/screen_energies.py --run checkpoints/test/<run> --epoch 19
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_mechanism import cos_curv, reach_terms, sample_windows, spearman, stack, tail_share  # noqa: E402


def recurrence(V, gamma=0.95, tau=1e-4):
    V = torch.cat([torch.zeros_like(V[:, :1]), V], dim=1)
    return F.relu(V[:, 1:] - gamma * V[:, :-1] - tau).mean(-1)


def tube_dt(h, anchor_eps=1e-3, eps=1e-8):
    c = h[:, -1] - h[:, 0]
    L_sq = (c ** 2).sum(-1).clamp(min=eps)
    c_hat = c / L_sq.sqrt().unsqueeze(-1)
    d = h[:, 1:-1] - h[:, :1]
    p = (d * c_hat.unsqueeze(1)).sum(-1)
    e_sq = ((d - p.unsqueeze(-1) * c_hat.unsqueeze(1)) ** 2).sum(-1)
    V = e_sq / (anchor_eps * L_sq.unsqueeze(1) + (d ** 2).sum(-1))
    return recurrence(V)


def heading(h, eps=1e-6):
    v = h[:, 1:] - h[:, :-1]
    c = (h[:, -1] - h[:, 0]).unsqueeze(1)
    V = 1 - F.cosine_similarity(v, c.expand_as(v), dim=-1, eps=eps)
    return recurrence(V)


def heading_goal(h, eps=1e-6):
    v = h[:, 1:-1] - h[:, :-2]
    g = h[:, -1:] - h[:, :-2]
    V = 1 - F.cosine_similarity(v, g.expand_as(v), dim=-1, eps=eps)
    return recurrence(V)


CANDIDATES = {
    "aggcos": lambda h: cos_curv(h),
    "reach_L": lambda h: reach_terms(h)[0],
    "tube_dt": tube_dt,
    "heading_chord": heading,
    "heading_goal": heading_goal,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--n_windows", type=int, default=512)
    ap.add_argument("--n_grad_batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

    wins = sample_windows(dset, args.n_windows, T, fs, rng)
    H, XY = [], []
    with torch.no_grad():
        for i in range(0, len(wins), args.batch):
            obs, act, xy = stack(wins[i : i + args.batch], device)
            zv = model.encode_obs(obs)["visual"].float()
            b, t, p, d = zv.shape
            H.append(enc.agg(zv.reshape(b * t, p, d)).reshape(b, t, -1).cpu())
            XY.append(xy)
    H, XY = torch.cat(H), torch.cat(XY)
    phys = cos_curv(XY.double()).float()
    for name, fn in CANDIDATES.items():
        lw = fn(H)
        out[f"{name}_mean"] = lw.mean().item()
        out[f"{name}_spearman_physcurv"] = spearman(lw[None], phys[None]).item()
        out[f"{name}_top10_share"] = tail_share(lw)

    proj = [p_ for n, p_ in enc.named_parameters() if "projector" in n]
    for p_ in proj:
        p_.requires_grad_(True)
    names = ["jepa"] + list(CANDIDATES)
    G = {k: None for k in names}
    gwins = sample_windows(dset, args.n_grad_batches * args.batch, T, fs, rng)
    for i in range(0, len(gwins), args.batch):
        obs, act, _ = stack(gwins[i : i + args.batch], device)
        z = model.encode(obs, act)
        z_pred = model.predict(z[:, : model.num_hist])
        z_tgt = z[:, model.num_pred :].detach()
        losses = {"jepa": F.mse_loss(z_pred[..., : -model.action_dim], z_tgt[..., : -model.action_dim])}
        zv = model.visual_only(z)
        b, t, p, d = zv.shape
        h = enc.agg(zv.reshape(b * t, p, d)).reshape(b, t, -1).float()
        for name, fn in CANDIDATES.items():
            losses[name] = fn(h).mean()
        for name, loss in losses.items():
            gs = torch.autograd.grad(loss, proj, retain_graph=True, allow_unused=True)
            flat = torch.cat([(g if g is not None else torch.zeros_like(p_)).flatten() for g, p_ in zip(gs, proj)])
            G[name] = flat if G[name] is None else G[name] + flat
    for name in CANDIDATES:
        out[f"{name}_grad_ratio_0.1_over_jepa"] = 0.1 * G[name].norm().item() / G["jepa"].norm().item()
        out[f"{name}_gradcos_aggcos"] = F.cosine_similarity(G[name], G["aggcos"], dim=0).item()
        out[f"{name}_gradcos_jepa"] = F.cosine_similarity(G[name], G["jepa"], dim=0).item()
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
