#!/usr/bin/env bash
# Matched lambda=0 control (dino_channel, same seed/lr/epochs): starts on GPU 2 once lyapreach1e-2 (pid 2141539) exits.
while kill -0 2141539 2>/dev/null; do sleep 120; done
cd /project/khanhnt/JEPA/temporal-straightening
export DATASET_DIR=/project/khanhnt/temporal-straightening/data PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2 PATH=/opt/anaconda/envs/ts/bin:$PATH
python train.py --config-name train.yaml env=point_maze encoder=dino_channel training.straighten=False > logs/lambda0_20261014_163231/lambda0.log 2>&1
