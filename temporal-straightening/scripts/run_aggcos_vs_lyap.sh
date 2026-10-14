#!/usr/bin/env bash
# Train the aggcos baseline and agg-Lyapunov variants in parallel, one GPU each.
#
#   conda activate ts
#   GPUS="1 2" LYAP_RUNS="lyapreach1e-1" bash scripts/run_aggcos_vs_lyap.sh [extra hydra overrides...]
#
# The first GPU runs the aggcos baseline; each LYAP_RUNS entry takes the next GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

export DATASET_DIR="${DATASET_DIR:-/project/khanhnt/temporal-straightening/data}"
export PYTHONNOUSERSITE=1
ENV="${ENV:-point_maze}"
BASELINE="${BASELINE:-aggcos1e-1}"
LYAP_RUNS="${LYAP_RUNS:-lyapreach1e-2}"
read -r -a GPU_LIST <<< "${GPUS:-2}"
RUNS=($LYAP_RUNS) #"$BASELINE"

if (( ${#RUNS[@]} > ${#GPU_LIST[@]} )); then
  echo "Need ${#RUNS[@]} GPUs for runs: ${RUNS[*]} (got GPUS=\"${GPU_LIST[*]}\")" >&2
  exit 1
fi

LOG_DIR="logs/aggcos_vs_lyap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

pids=()
for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  gpu="${GPU_LIST[$i]}"
  echo "[gpu $gpu] training.straighten=$run -> $LOG_DIR/$run.log"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py --config-name train.yaml \
    env="$ENV" encoder=dino_channel training.straighten="$run" "$@" \
    > "$LOG_DIR/$run.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
