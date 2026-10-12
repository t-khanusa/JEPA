"""Scale-aware rollout diagnostics for a trained checkpoint (eval only).

Raw latent MSE is not comparable across models whose feature scale differs, so
every error is also reported relative to the model's own between-state spread.

  python scripts/diagnose_scale.py --run checkpoints/test/<run> --epoch 8 [--repo <code dir>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


def find_attr(obj, name):
    while obj is not None:
        if hasattr(obj, name):
            return getattr(obj, name)
        obj = getattr(obj, "dataset", None)
    raise AttributeError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
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

    fs, H = cfg.frameskip, args.horizon
    a_mean = find_attr(dset, "action_mean").to(device)
    a_std = find_attr(dset, "action_std").to(device)
    rng = np.random.default_rng(args.seed)

    windows = []
    while len(windows) < args.n:
        obs, act, _, _ = dset[int(rng.integers(len(dset)))]
        T = obs["visual"].shape[0]
        if T <= H * fs + 1:
            continue
        s = int(rng.integers(0, T - H * fs - 1))
        o = {k: v[s : s + H * fs + 1 : fs] for k, v in obs.items()}
        a = rearrange(act[s : s + H * fs], "(h f) d -> h (f d)", f=fs)
        windows.append((o, a))

    acc = {k: [] for k in ["z_true", "z_demo", "z_zero"]}
    with torch.no_grad():
        for i in range(0, args.n, args.batch):
            chunk = windows[i : i + args.batch]
            obs = {k: torch.stack([w[0][k] for w in chunk]).to(device) for k in chunk[0][0]}
            act = torch.stack([w[1] for w in chunk]).to(device)
            z_true = model.encode_obs(obs)["visual"].float()
            obs_0 = {k: v[:, :1] for k, v in obs.items()}
            z_demo = model.rollout(obs_0, act)[0]["visual"].float()
            zero = ((0.0 - a_mean) / a_std).repeat(fs)
            z_zero = model.rollout(obs_0, zero.expand_as(act).contiguous())[0]["visual"].float()
            acc["z_true"].append(z_true.cpu())
            acc["z_demo"].append(z_demo.cpu())
            acc["z_zero"].append(z_zero.cpu())
    z_true, z_demo, z_zero = (torch.cat(acc[k]) for k in ["z_true", "z_demo", "z_zero"])

    def mse(a, b):
        return ((a - b) ** 2).flatten(1).mean(1)

    flat = z_true.flatten(2)
    var = flat.reshape(-1, flat.shape[-1]).var(dim=0).mean().item()
    step_mse = mse(z_true[:, 1:].flatten(0, 1), z_true[:, :-1].flatten(0, 1)).mean().item()
    start_goal = mse(z_true[:, 0], z_true[:, -1])
    demo_goal = mse(z_demo[:, -1], z_true[:, -1])
    zero_goal = mse(z_zero[:, -1], z_true[:, -1])
    per_step = [mse(z_demo[:, k], z_true[:, k]).mean().item() for k in range(1, H + 1)]

    out = {
        "run": run.name,
        "epoch": args.epoch,
        "feature_var": var,
        "step_mse": step_mse,
        "raw_rollout_mse_H": demo_goal.mean().item(),
        "nmse_rollout_H": demo_goal.mean().item() / var,
        "rollout_mse_over_step": demo_goal.mean().item() / step_mse,
        "rollout_mse_per_step_over_step": [e / step_mse for e in per_step],
        "imagined_demo_cost_ratio": (demo_goal / start_goal.clamp(min=1e-12)).median().item(),
        "demo_over_zero_cost": (demo_goal / zero_goal.clamp(min=1e-12)).median().item(),
        "demo_beats_zero_rate": (demo_goal < zero_goal).float().mean().item(),
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
