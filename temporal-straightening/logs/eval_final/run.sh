#!/usr/bin/env bash
# One queue per GPU (job i -> GPU i%3), at most 4 concurrent jobs per GPU (~5 GB each).
cd /project/khanhnt/JEPA/temporal-straightening
D=logs/eval_final
for g in 0 1 2; do
  awk -v g=$g '(NR-1)%3==g {print g, $0}' $D/jobs.txt | xargs -P 4 -L 1 bash $D/worker.sh &
done
wait
echo ALL_DONE
