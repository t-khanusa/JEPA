#!/usr/bin/env bash
# Planning-objective sweep (stock last vs Lyapunov objectives) on paired episodes.
#   GPU=1 RUN=<ckpt dir name> bash scripts/run_plan_lyap_sweep.sh

cd "$(dirname "$0")/.."
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1 || true
set -eo pipefail
export PYTHONNOUSERSITE=1 DATASET_DIR="${DATASET_DIR:-/project/khanhnt/temporal-straightening/data}" WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
CKPT_BASE="${CKPT_BASE:-checkpoints/test}"
OBJECTIVES="${OBJECTIVES:-last reach_agg_w1 reach_agg_w0.1 heading_agg_w1 heading_agg_w0.1 reach_patch_w1 heading_patch_w1 cost_w1}"
/opt/anaconda/envs/ts/bin/python scripts/plan_lyap_sweep.py --run "$CKPT_BASE/$RUN" --epoch "${EPOCH:-20}" \
  --objectives $OBJECTIVES
