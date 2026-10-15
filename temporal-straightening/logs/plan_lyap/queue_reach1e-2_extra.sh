#!/usr/bin/env bash
# After queue_reach1e-2.sh (pid 2508152): rerun `last` (nondeterminism check) + planner-side Lyapunov objectives for lyapreach1e-2 on GPU 1.
while kill -0 2508152 2>/dev/null; do sleep 30; done
cd /project/khanhnt/JEPA/temporal-straightening
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
export PYTHONNOUSERSITE=1 DATASET_DIR=/project/khanhnt/temporal-straightening/data CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline
R=checkpoints/test/umaze_lyapreach1e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
P=/opt/anaconda/envs/ts/bin/python
$P scripts/plan_lyap_sweep.py --run $R --epoch 20 --objectives last --out logs/plan_lyap_rerun > logs/plan_lyap/reach1e-2_rerun.out 2>&1
$P scripts/plan_lyap_sweep.py --run $R --epoch 20 --objectives reach_agg_w0.1 heading_agg_w0.1 reach_patch_w1 heading_patch_w1 > logs/plan_lyap/reach1e-2_obj.out 2>&1
