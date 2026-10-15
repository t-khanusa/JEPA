#!/usr/bin/env bash
# After matched lambda=0 training (pid 2553712) exits: paired planning + scale/trainability diagnostics on GPU 2.
while kill -0 2553712 2>/dev/null; do sleep 120; done
cd /project/khanhnt/JEPA/temporal-straightening
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
export PYTHONNOUSERSITE=1 DATASET_DIR=/project/khanhnt/temporal-straightening/data CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline
R=checkpoints/test/umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
P=/opt/anaconda/envs/ts/bin/python
$P scripts/plan_lyap_sweep.py --run $R --epoch 20 --objectives last > logs/plan_lyap/lambda0.out 2>&1
for e in 4 10 20; do $P scripts/diagnose_scale.py --run $R --epoch $e > logs/diagnostics_e20/umaze_False_scale_e$e.log 2>&1; done
$P scripts/diagnose_trainability.py --run $R --epoch 20 > logs/diag_trainability/lambda0_e20.log 2>&1
