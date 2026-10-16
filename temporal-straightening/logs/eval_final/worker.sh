#!/usr/bin/env bash
# worker.sh <gpu> <kind> <run> <epoch>
cd /project/khanhnt/JEPA/temporal-straightening
export PATH=/opt/anaconda/envs/ts/bin:$PATH WANDB_MODE=offline
source /project/khanhnt/temporal-straightening/setup_env.sh >/dev/null 2>&1
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$1
D=logs/eval_final
if [ "$2" = plan ]; then
  python scripts/plan_lyap_sweep.py --run checkpoints/test/$3 --epoch $4 --objectives last_raw --out logs/plan_lyap_epochs > $D/plan_$3_e$4.out 2>&1
else
  python scripts/diagnose_scale.py --run checkpoints/test/$3 --epoch $4 > $D/scale_$3_e$4.log 2>&1
fi
echo "$(date +%T) done $2 $3 e$4 rc=$?"
