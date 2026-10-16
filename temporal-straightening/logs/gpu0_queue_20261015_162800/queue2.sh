#!/usr/bin/env bash
# Continue GPU0 queue: wait for the running e8 reference planning, run e16, then train Reach lambda=0.3 (lr 1e-5).
cd /project/khanhnt/JEPA/temporal-straightening
export PATH=/opt/anaconda/envs/ts/bin:$PATH PYTHONNOUSERSITE=1 WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
D=logs/gpu0_queue_20261015_162800
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 >> $D/gpu0_mem.log; }
while kill -0 3406677 3406678 3406680 2>/dev/null; do mem; sleep 20; done
pids=()
for r in aggmlpcos1e-1 lyapreach1e-1 False; do
  python scripts/plan_lyap_sweep.py --run checkpoints/test/umaze_${r}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 \
    --epoch 16 --objectives last_raw --out logs/plan_lyap_epochs > $D/plan_${r}_e16.out 2>&1 &
  pids+=($!)
done
while kill -0 "${pids[@]}" 2>/dev/null; do mem; sleep 20; done
GPUS="0" LYAP_RUNS="lyapreach3e-1" bash scripts/run_aggcos_vs_lyap.sh > $D/train_reach3e-1.out 2>&1
