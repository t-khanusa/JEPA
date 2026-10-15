#!/usr/bin/env bash
# Stock-objective planning (last_raw) at intermediate epochs: does early planning success predict epoch 20?
#   GPU=<n> RUNS="<short names>" EPOCHS="4 10" bash logs/plan_lyap/run_epoch_planning.sh
cd /project/khanhnt/JEPA/temporal-straightening
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
export PYTHONNOUSERSITE=1 DATASET_DIR=/project/khanhnt/temporal-straightening/data CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline
P=/opt/anaconda/envs/ts/bin/python
for e in $EPOCHS; do for r in $RUNS; do
  R=checkpoints/test/umaze_${r}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
  $P scripts/plan_lyap_sweep.py --run $R --epoch $e --objectives last_raw --out logs/plan_lyap_epochs > logs/plan_lyap_epochs_${r}_e$e.out 2>&1
  $P scripts/diagnose_scale.py --run $R --epoch $e > logs/plan_lyap_epochs_${r}_e${e}_scale.log 2>&1
done; done
