#!/usr/bin/env bash
# Pre-report evaluation while all 3 GPUs train (planning needs ~5 GB, so it cannot share a GPU with training).
#  A) when both lr1e-6 runs start epoch 17: stop them, plan e8/e16 (last_raw) + T6 e8, scale diagnostics.
#  B) when the T6 run starts epoch 17: plan T6 e16 + its scale diagnostics.
#  C) resume the lr1e-6 runs from model_16 for the remaining 4 epochs.
cd /project/khanhnt/JEPA/temporal-straightening
export PATH=/opt/anaconda/envs/ts/bin:$PATH WANDB_MODE=offline
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
export PYTHONNOUSERSITE=1
D=logs/eval_before_report
CK=checkpoints/test
L0=umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
R6=umaze_lyapreach1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
T6=umaze_lyapreach1e-1T6_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
LOG6=logs/aggcos_vs_lyap_20261015_154532
LOGT6=logs/aggcos_vs_lyap_20261015_171906/lyapreach1e-1.log
started() { grep -qa "Epoch $2 decoder_active" "$1"; }
plan() { CUDA_VISIBLE_DEVICES=$1 python scripts/plan_lyap_sweep.py --run $CK/$2 --epoch $3 --objectives last_raw \
           --out logs/plan_lyap_epochs > $D/plan_$2_e$3.out 2>&1; }
scale() { CUDA_VISIBLE_DEVICES=$1 python scripts/diagnose_scale.py --run $CK/$2 --epoch $3 > $D/scale_$2_e$3.log 2>&1; }

until started $LOG6/false.log 17 && started $LOG6/lyapreach1e-1.log 17; do sleep 30; done
echo "$(date) A: stopping lr1e-6 runs"
pkill -f "training.encoder_lr=1e-6"; sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
( plan 1 $L0 8 & plan 1 $L0 16 & plan 1 $T6 8 & wait; scale 1 $L0 16; scale 1 $L0 8 ) &
( plan 2 $R6 8 & plan 2 $R6 16 & wait; scale 2 $R6 16; scale 2 $R6 8
  for r in umaze_aggmlpcos1e-1 umaze_lyapreach1e-1 umaze_False; do scale 2 ${r}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 16; done ) &
wait
echo "$(date) A done"
until started $LOGT6 17; do sleep 30; done
echo "$(date) B: T6 e16"
plan 1 $T6 16 & scale 2 $T6 16 & scale 2 $T6 8 & wait
echo "$(date) B done; C: resuming lr1e-6 runs"
for r in $L0 $R6; do cp $CK/$r/checkpoints/model_16.pth $CK/$r/checkpoints/model_latest.pth; rm -f $CK/$r/checkpoints/model_17.pth; done
GPUS="1 2" LYAP_RUNS="false lyapreach1e-1" bash scripts/run_aggcos_vs_lyap.sh training.encoder_lr=1e-6 training.epochs=4 > $D/resume.out 2>&1
