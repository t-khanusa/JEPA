"""Does the straightening gradient actually improve anything? (eval only, no weights saved)

On training-length windows (T = num_hist + num_pred) of a checkpoint, for aggcos / reach /
heading on agg latents and for the JEPA z_loss:

A. Recurrence credit (reach, heading): d loss / d V_t per transition, share of windows where
   some V_t gets a NEGATIVE weight (loss decreases when that transition gets worse).
B. Gradient coherence over encoder trainable params (projector + agg head): mean pairwise
   cosine between minibatch gradients and SNR = ||mean g|| / mean ||g_i||. A large but
   incoherent gradient changes the weights without a consistent direction.
C. One-step probe: move the trainable encoder params along -g/||g|| of each loss (g from set A,
   step = eta * ||theta||), then measure the relative change of every loss on held-out set B.
   Diagonal = can this loss be decreased at all here; off-diagonal = side effects (e.g. on JEPA).

  python scripts/diagnose_trainability.py --run checkpoints/test/<run> --epoch 20
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def recurrence_weights(V, gamma, anchor_zero):
    """Exact d(sum relu(V_{k+1} - gamma V_k - tau)) / dV for each window: (B, K)."""
    V = V.detach().clone().requires_grad_(True)
    if anchor_zero:
        prev = torch.cat([torch.zeros_like(V[:, :1]), V[:, :-1]], 1)
        viol = V - gamma * prev - 1e-4
    else:
        viol = V[:, 1:] - gamma * V[:, :-1] - 1e-4
    F.relu(viol).sum().backward()
    return V.grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", default="20")
    ap.add_argument("--n_windows", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--etas", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import hydra
    from omegaconf import OmegaConf
    import custom_resolvers  # noqa: F401
    from plan import load_model
    from diagnose_mechanism import sample_windows, stack
    from models.lyapunov import TSLyapunovRegularizer

    torch.manual_seed(args.seed)
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
    T = cfg.num_hist + cfg.num_pred
    rng = np.random.default_rng(args.seed)
    wins = sample_windows(traj["valid"], args.n_windows, T, cfg.frameskip, rng)
    half = len(wins) // 2
    set_A, set_B = wins[:half], wins[half:]

    reach = TSLyapunovRegularizer(mode="lyapreach")
    heading = TSLyapunovRegularizer(mode="lyapheading")

    def cos_loss(h):
        v = h[:, 1:] - h[:, :-1]
        return (1 - F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1, eps=1e-6)).mean()

    def losses(chunk, keys):
        obs, act, _ = stack(chunk, device)
        z = model.encode(obs, act)
        zv = model.visual_only(z)
        b, t, p, d = zv.shape
        h = enc.agg(zv.reshape(b * t, p, d)).reshape(b, t, -1).float()
        out = {}
        if "jepa" in keys:
            z_pred = model.predict(z[:, : model.num_hist])
            tgt = z[:, model.num_pred:].detach()
            out["jepa"] = F.mse_loss(z_pred[..., : -model.action_dim], tgt[..., : -model.action_dim])
        if "aggcos" in keys:
            out["aggcos"] = cos_loss(h)
        if "reach" in keys:
            out["reach"] = reach(h)[0]
        if "heading" in keys:
            out["heading"] = heading(h)[0]
        return out, h

    keys = ["jepa", "aggcos", "reach", "heading"]
    res = {"run": run.name, "epoch": args.epoch}

    # A. recurrence credit on agg latents
    with torch.no_grad():
        H = torch.cat([losses(set_A[i:i + args.batch], [])[1] for i in range(0, half, args.batch)])
    from planning.lyapunov_objectives import reach_energy, heading_energy
    V_r = reach_energy(H, H[:, -1])
    V_h = heading_energy(H, H[:, -1])
    for name, V, anchor in (("reach", V_r, False), ("heading", V_h, True)):
        w = recurrence_weights(V, 0.95, anchor)
        res[f"A_{name}_dV_mean_per_t"] = [round(x, 4) for x in w.mean(0).tolist()]
        res[f"A_{name}_neg_window_frac"] = (w < 0).any(1).float().mean().item()
        res[f"A_{name}_zero_grad_window_frac"] = (w.abs().sum(1) == 0).float().mean().item()

    # B. gradient coherence over minibatches
    grads = {k: [] for k in keys}
    for i in range(0, half, args.batch):
        out, _ = losses(set_A[i:i + args.batch], keys)
        for k in keys:
            g = torch.autograd.grad(out[k], params, retain_graph=True, allow_unused=True)
            grads[k].append(torch.cat([(x if x is not None else torch.zeros_like(p)).flatten()
                                       for x, p in zip(g, params)]).float())
    for k in keys:
        G = torch.stack(grads[k])
        Gn = F.normalize(G, dim=1)
        C = Gn @ Gn.T
        n = G.shape[0]
        res[f"B_{k}_pairwise_cos"] = ((C.sum() - n) / (n * (n - 1))).item()
        res[f"B_{k}_snr"] = (G.mean(0).norm() / G.norm(dim=1).mean()).item()
        res[f"B_{k}_gradnorm"] = G.norm(dim=1).mean().item()
    gbar = {k: torch.stack(grads[k]).mean(0) for k in keys}
    for a, b in itertools.combinations(keys, 2):
        res[f"B_cos_mean_{a}_{b}"] = F.cosine_similarity(gbar[a], gbar[b], dim=0).item()

    # C. one-step probe on held-out set B
    @torch.no_grad()
    def eval_B():
        acc = {k: 0.0 for k in keys}
        n = 0
        for i in range(0, len(set_B), args.batch):
            out, _ = losses(set_B[i:i + args.batch], keys)
            for k in keys:
                acc[k] += out[k].item()
            n += 1
        return {k: v / n for k, v in acc.items()}

    theta0 = torch.cat([p.detach().flatten() for p in params]).clone()
    base = eval_B()
    res["C_base"] = base
    for eta in args.etas:
        for k in keys:
            step = -gbar[k] / gbar[k].norm().clamp(min=1e-12) * eta * theta0.norm()
            torch.nn.utils.vector_to_parameters(theta0 + step, params)
            after = eval_B()
            res[f"C_eta{eta:g}_step_{k}"] = {m: (after[m] - base[m]) / abs(base[m]) for m in keys}
        torch.nn.utils.vector_to_parameters(theta0, params)

    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
