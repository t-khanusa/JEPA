"""Compare planning objectives (stock `last` vs Lyapunov Reach / Heading / cost-CLF) on paired episodes.

Loads the model, val set and envs once, then for every planning seed builds the same
targets as `plan.py --config-name plan_gd.yaml seed=<s> n_evals=<n>` and runs every
objective from a fresh planner on them, so success is paired across objectives and with
earlier plan.py runs. Results go to <out>/<run>.jsonl, one line per (seed, objective).

  python scripts/plan_lyap_sweep.py --run checkpoints/test/<run> --epoch 20 \
      --objectives last heading_agg_w1 reach_agg_w1 cost_w1
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def make_planner(ws, planner_cfg, objective_fn):
    """Same instantiation as PlanWorkspace.__init__, with a fresh MPC state."""
    import hydra

    planner_cfg = copy.deepcopy(planner_cfg)
    if "n_taken_actions" in planner_cfg:
        planner_cfg["n_taken_actions"] = planner_cfg["n_taken_actions"] // ws.frameskip
    if "sub_planner" in planner_cfg and "horizon" in planner_cfg["sub_planner"]:
        planner_cfg["sub_planner"]["horizon"] = planner_cfg["sub_planner"]["horizon"] // ws.frameskip
    planner = hydra.utils.instantiate(
        planner_cfg,
        wm=ws.wm,
        env=ws.env,
        action_dim=ws.action_dim,
        objective_fn=objective_fn,
        preprocessor=ws.data_preprocessor,
        evaluator=ws.evaluator,
        wandb_run=ws.wandb_run,
        log_filename=ws.log_filename,
    )
    planner.horizon = ws.goal_H_model_steps
    return planner


@torch.no_grad()
def imagined_terms(ws, actions, objective_fn):
    from utils import move_to_device

    obs_0 = move_to_device(ws.data_preprocessor.transform_obs(ws.obs_0), ws.device)
    obs_g = move_to_device(ws.data_preprocessor.transform_obs(ws.obs_g), ws.device)
    z_obs, _ = ws.wm.rollout(obs_0=obs_0, act=actions)
    z_g = ws.wm.encode_obs(obs_g)
    _, terms = objective_fn(z_obs, z_g, return_terms=True)
    return {k: v.float().cpu().tolist() for k, v in terms.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", default="20")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 1001, 100)))
    ap.add_argument("--n_evals", type=int, default=5)
    ap.add_argument("--objectives", nargs="+", default=["last", "heading_agg_w1", "reach_agg_w1"])
    ap.add_argument("--plan_cfg", default=str(REPO / "conf" / "plan_gd.yaml"))
    ap.add_argument("--out", default=str(REPO / "logs" / "plan_lyap"))
    args = ap.parse_args()

    import gym
    import hydra
    from omegaconf import OmegaConf
    import custom_resolvers  # noqa: F401
    from plan import PlanWorkspace, DummyWandbRun, load_model
    from planning.lyapunov_objectives import create_lyapunov_objective_fn, parse_objective_spec
    from env.serial_vector_env import SerialVectorEnv
    from utils import seed as seed_all

    run = Path(args.run).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run.name}_e{args.epoch}.jsonl"
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["objective"]))

    model_cfg = OmegaConf.load(run / "hydra.yaml")
    plan_cfg = OmegaConf.load(args.plan_cfg)
    for k in ("defaults", "hydra"):
        plan_cfg.pop(k, None)
    base_cfg = OmegaConf.to_container(plan_cfg)
    base_cfg.update(
        n_evals=args.n_evals, model_name=run.name, model_epoch=args.epoch,
        decode_for_viz=False, wandb_logging=False,
    )
    pristine_planner_cfg = copy.deepcopy(base_cfg["planner"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, dset = hydra.utils.call(
        model_cfg.env.dataset, num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]
    wm = load_model(run / "checkpoints" / f"model_{args.epoch}.pth", model_cfg,
                    model_cfg.num_action_repeat, device)
    wm.eval()
    env = SerialVectorEnv([
        gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
        for _ in range(args.n_evals)
    ])

    for s in args.seeds:
        todo = [o for o in args.objectives if (s, o) not in done]
        if not todo:
            continue
        work = out_dir / run.name / f"s{s}"
        work.mkdir(parents=True, exist_ok=True)
        os.chdir(work)
        cfg_dict = copy.deepcopy(base_cfg)
        cfg_dict.update(seed=s, saved_folder=str(work))
        seed_all(s)  # plan.py seeds right before building targets as well
        ws = PlanWorkspace(cfg_dict=cfg_dict, wm=wm, dset=dset, env=env,
                           env_name=model_cfg.env.name, frameskip=model_cfg.frameskip,
                           wandb_run=DummyWandbRun())
        for obj in todo:
            t0 = time.perf_counter()
            kw = parse_objective_spec(obj)
            objective_fn = create_lyapunov_objective_fn(wm=wm, alpha=base_cfg["objective"]["alpha"], **kw)
            seed_all(s)
            planner = make_planner(ws, pristine_planner_cfg, objective_fn)
            actions, action_len = planner.plan(obs_0=ws.obs_0, obs_g=ws.obs_g, actions=None)
            logs, successes, _, e_states = ws.evaluator.eval_actions(
                actions.detach(), action_len, save_video=False, filename=f"final_{obj}")
            state_dist = ws.env.eval_state(ws.state_g, e_states[:, -1]).get("state_dist", [])
            rec = {
                "run": run.name, "epoch": args.epoch, "seed": s, "objective": obj,
                "eval_seed": ws.eval_seed,
                "success": [bool(x) for x in successes],
                "state_dist": [float(x) for x in np.asarray(state_dist)],
                "terms": imagined_terms(ws, actions.detach(), objective_fn),
                "secs": round(time.perf_counter() - t0, 1),
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{run.name}] seed={s} obj={obj} success={sum(rec['success'])}/{len(rec['success'])} "
                  f"({rec['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
