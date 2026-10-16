#!/usr/bin/env bash
# GPU0: wait for e8 reference planning, run e16 reference planning, then train Reach 0.1 with a 6-frame geometry window.
cd /project/khanhnt/JEPA/temporal-straightening
export PATH=/opt/anaconda/envs/ts/bin:$PATH WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
export PYTHONNOUSERSITE=1
D=logs/gpu0_queue_20261015_162800
while kill -0 3406677 3406678 3406680 2>/dev/null; do sleep 15; done
pids=()
for r in aggmlpcos1e-1 lyapreach1e-1 False; do
  python scripts/plan_lyap_sweep.py --run checkpoints/test/umaze_${r}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 \
    --epoch 16 --objectives last_raw --out logs/plan_lyap_epochs > $D/plan_${r}_e16.out 2>&1 &
  pids+=($!)
done
while kill -0 "${pids[@]}" 2>/dev/null; do sleep 15; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS="0" LYAP_RUNS="lyapreach1e-1" bash scripts/run_aggcos_vs_lyap.sh training.lyap_frames=6 > $D/train_reach1e-1_T6.out 2>&1
