"""Offline screen of Lyapunov-loss variants by gradient coherence (eval only, no training).

For each variant, gradients w.r.t. the trainable encoder params (projector + agg head) on
disjoint minibatches of val windows: mean pairwise cosine (does the loss push consistently?)
and cosine to the JEPA gradient. Window length --frames can exceed the training window.

  python scripts/screen_lyap_variants.py --run checkpoints/test/<run> --frames 4 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

VARIANTS = {
    "aggcos": dict(kind="aggcos"),
    "reach": dict(kind="lyapreach"),
    "reach_sp": dict(kind="lyapreach", use_softplus=True),
    "reach_g0": dict(kind="lyapreach", gamma=0.0),
    "heading": dict(kind="lyapheading"),
    "heading_sp": dict(kind="lyapheading", use_softplus=True),
    "heading_g0": dict(kind="lyapheading", gamma=0.0),
    "heading_g0.5": dict(kind="lyapheading", gamma=0.5),
    "heading_EpD": [dict(kind="lyapheading", gamma=0.0), dict(kind="lyapheading")],  # energy + decrease
    "reach_EpD": [dict(kind="lyapreach", gamma=0.0), dict(kind="lyapreach")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", default="20")
    ap.add_argument("--frames", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n_windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import hydra
    from omegaconf import OmegaConf
    import custom_resolvers  # noqa: F401
    from plan import load_model
    from diagnose_mechanism import sample_windows, stack
    from models.lyapunov import TSLyapunovRegularizer

    run = Path(args.run).resolve()
    cfg = OmegaConf.load(run / "hydra.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, traj = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
                               frameskip=cfg.frameskip)
    model = load_model(run / "checkpoints" / f"model_{args.epoch}.pth", cfg, cfg.num_action_repeat, device)
    model.eval()
    enc = model.encoder
    params = [p for n, p in enc.named_parameters() if not n.startswith("base_model")]
    for p in params:
        p.requires_grad_(True)

    def build(v):
        return TSLyapunovRegularizer(mode=v["kind"], **{a: b for a, b in v.items() if a != "kind"})

    regs = {k: [build(c) for c in (v if isinstance(v, list) else [v])]
            for k, v in VARIANTS.items() if k != "aggcos"}

    def flat(g):
        return torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(g, params)]).float()

    res = {"run": run.name, "epoch": args.epoch}
    for nf in args.frames:
        rng = np.random.default_rng(args.seed)
        wins = sample_windows(traj["valid"], args.n_windows, nf, cfg.frameskip, rng)
        grads = {k: [] for k in list(VARIANTS) + ["jepa"]}
        for i in range(0, len(wins), args.batch):
            obs, act, _ = stack(wins[i:i + args.batch], device)
            z = model.encode(obs, act)
            zv = model.visual_only(z)
            b, t, p, d = zv.shape
            h = enc.agg(zv.reshape(b * t, p, d)).reshape(b, t, -1).float()
            # JEPA on the training-length prefix, as in training
            zt = z[:, : cfg.num_hist + cfg.num_pred]
            z_pred = model.predict(zt[:, : model.num_hist])
            jepa = F.mse_loss(z_pred[..., : -model.action_dim], zt[:, model.num_pred:].detach()[..., : -model.action_dim])
            grads["jepa"].append(flat(torch.autograd.grad(jepa, params, retain_graph=True, allow_unused=True)))
            for k in VARIANTS:
                if k == "aggcos":
                    v = h[:, 1:] - h[:, :-1]
                    loss = (1 - F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1, eps=1e-6)).mean()
                else:
                    loss = sum(r(h)[0] for r in regs[k])
                grads[k].append(flat(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)))
        gj = torch.stack(grads["jepa"]).mean(0)
        for k in list(VARIANTS) + ["jepa"]:
            G = torch.stack(grads[k])
            Gn = F.normalize(G, dim=1)
            n = G.shape[0]
            res[f"T{nf}_{k}_pairwise_cos"] = round(((Gn @ Gn.T).sum().item() - n) / (n * (n - 1)), 4)
            res[f"T{nf}_{k}_cos_jepa"] = round(F.cosine_similarity(G.mean(0), gj, dim=0).item(), 4)
            res[f"T{nf}_{k}_gradnorm"] = round(G.norm(dim=1).mean().item(), 4)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
