#!/usr/bin/env bash
# After lyapreach1e-2 training (pid 2141539) exits: paired planning (stock objective) + scale/trainability diagnostics on GPU 1.
while kill -0 2141539 2>/dev/null; do sleep 120; done
cd /project/khanhnt/JEPA/temporal-straightening
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
export PYTHONNOUSERSITE=1 DATASET_DIR=/project/khanhnt/temporal-straightening/data CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline
R=checkpoints/test/umaze_lyapreach1e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
P=/opt/anaconda/envs/ts/bin/python
$P scripts/plan_lyap_sweep.py --run $R --epoch 20 --objectives last > logs/plan_lyap/reach1e-2.out 2>&1
$P scripts/diagnose_scale.py --run $R --epoch 20 > logs/diagnostics_e20/umaze_lyapreach1e-2_scale.log 2>&1
$P scripts/diagnose_trainability.py --run $R --epoch 20 > logs/diag_trainability/lyapreach1e-2_e20.log 2>&1
