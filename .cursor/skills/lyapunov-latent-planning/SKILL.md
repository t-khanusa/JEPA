---
name: lyapunov-latent-planning
description: Goal, design, hyperparameters and evaluation protocol for replacing the temporal-straightening (TS) aggcos curvature loss with a ControlJEPA Lyapunov loss for latent planning. Use when editing models/lyapunov/, the lyapreach/lyaptube straighten modes in VWorldModel, running aggcos-vs-Lyapunov experiments, or judging which run will plan better.
---

# Lyapunov loss for latent planning (ControlJEPA -> TS)

## Goal
Replace `training.straighten=aggcos1e-1` (cosine curvature on `encoder.agg()` latents)
with a Lyapunov tube/reach loss that yields a latent space where **gradient/CEM planning
succeeds more often**. The final judge is planning success (`plan.py`), not training loss.

Why it can beat cosine: cosine only constrains local turning angle between consecutive
steps (2 terms per 4-frame window), is scale-invariant and is blind to speed and to
progress toward the goal. The planner minimizes `||z_T - z_goal||^2`; that cost is
well-shaped when the goal distance along demos decreases monotonically and stays inside
a thin tube around the start->goal chord. The Lyapunov loss targets exactly that.

## Design (code: `temporal-straightening/models/lyapunov/`)
Per window h = agg(visual_only(z)) with shape (B, T=num_hist+num_pred=4, d=128):
- Chord v_geo = h_{T-1} - h_0, L = ||v_geo||; d_t = h_t - h_0; p_t = <d_t, v?>; e_t = d_t - p_t v?.
- `lyaptube`: V_t = ||e_t||^2 / L^2, loss = mean softplus/ReLU(V_{t+1} - ? V_t - ?). ISS tube bound V <= ?/(1-?).
- `lyapreach` (default): V_t = ? ||e_t||^2/L^2 + ? max(0, ?_t - p_t/L)^2 with ?_t = t/(T-1)
  (lag-only, schedule_asym). No pressure when ahead of schedule, so it cannot teleport.
- Optional rollout tube (`rollout_weight>0`, needs num_pred=1): tube on
  [h_0, agg(?_1..?_{T-2}), sg(h_{T-1})] from teacher-forced predictor outputs. This shapes the
  predictor, not only the encoder.
- Windows with L^2 < min_L_sq (static frames) are dropped.

Files:
- `control_losses.py`: vendored `lyapunov_controlJEPA.py` (unchanged).
- `planning_losses.py`: vendored `lyapunov_planning.py`, import fixed to `.control_losses`.
- `ts_lyapunov.py`: `TSLyapunovRegularizer` + `trajectory_geometry_metrics`.
- `models/visual_world_model.py`: `straighten` prefix `lyapreach<?>` / `lyaptube<?>`; loss added as
  `? * L_lyap`, logged as `lyapunov_loss_used_for_training`.
- `conf/train.yaml` -> `training.lyapunov` block.

Known quirks:
- `LyapunovReachabilityLoss` hardcodes `self.use_softplus = False` (always ReLU).
- With T=4 there are only 2 transitions per window. A longer window (num_hist up, or a dedicated
  longer clip just for the regularizer) is the biggest lever if the short-window result is flat.
- `z_visual_err_next1` in `train.py` slices `[-1:0]`, which is empty when num_pred=1. Ignore that metric.

## Hyperparameters (start / sweep)
| param | start | sweep |
|---|---|---|
| ? (suffix) | 1e0 | 1e-1, 1e0, 1e1 (scale differs from cosine; do not reuse 1e-1 blindly) |
| ? | 0.95 | 0.9, 0.99 |
| ? | 1e-4 | 1e-5, 1e-3 |
| ? / ? | 1.0 / 0.5 | ? in {0, 0.5, 1}; ?=0 is ? the pure tube |
| rollout_weight | 0 | 0.5 after the demo-only run works |
Keep everything else identical to the aggcos baseline (encoder=dino_channel, seed, lr, epochs).

## Metrics (in priority order)
1. **Planning success rate** (`plan.py`, same planner/seed/goal set). This is the only final verdict.
2. `val_z_visual_err_rollout_1framestart` / `val_z_visual_err_rollout`: open-loop multi-step error.
   This is the best training-time predictor of planning.
3. Geometry, logged identically in BOTH runs (`train_/val_geo_*`):
   - `geo_goal_monotone_rate`: fraction of steps where ||h_{t+1}-h_T|| < ||h_t-h_T||. Target ? 1.
   - `geo_V_transverse_mean/max`: tube thickness. Lower is better.
   - `geo_curvature` (=1-cos): lets Lyapunov runs be compared on aggcos's own metric.
   - `geo_progress_monotone_rate`, `geo_chord_over_path` (?1 means straight and efficient).
   - `geo_latent_std`: collapse guard. If it decays toward 0, the regularizer is winning by collapsing.
4. Lyapunov internals: `lyap_viol_rate`, `lyap_lag_rate`, `lyap_mean_V*`, `lyap_num_valid`.
5. `val_z_loss`: must not degrade noticeably versus baseline. Never compare total `val_loss` across
   runs, because it sums different regularizers.

Decision rule: prefer the run with lower rollout error AND higher goal-monotone rate at matched
`geo_latent_std`; confirm with planning success before claiming a win.

## Running
```bash
cd temporal-straightening
DATASET_DIR=/path ENV=point_maze GPUS="0 1 2" LYAP_RUNS="lyapreach1e0 lyaptube1e0" \
  bash scripts/run_aggcos_vs_lyap.sh
```
Hydra output dirs differ by the `straighten` string, so the runs do not collide.
