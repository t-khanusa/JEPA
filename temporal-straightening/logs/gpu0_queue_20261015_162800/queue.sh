#!/usr/bin/env bash
# GPU0: matched-epoch reference planning (lr1e-5 runs, e8/e16, last_raw), then Reach lambda=0.3 training at lr 1e-5.
cd /project/khanhnt/JEPA/temporal-straightening
export PATH=/opt/anaconda/envs/ts/bin:$PATH PYTHONNOUSERSITE=1 WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
D="$1"
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0; done ) > $D/gpu0_mem.log &
MEMPID=$!
for ep in 8 16; do
  for r in aggmlpcos1e-1 lyapreach1e-1 False; do
    python scripts/plan_lyap_sweep.py --run checkpoints/test/umaze_${r}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 \
      --epoch $ep --objectives last_raw --out logs/plan_lyap_epochs > $D/plan_${r}_e${ep}.out 2>&1 &
  done
  wait
done
kill $MEMPID
GPUS="0" LYAP_RUNS="lyapreach3e-1" bash scripts/run_aggcos_vs_lyap.sh > $D/train_reach3e-1.out 2>&1
