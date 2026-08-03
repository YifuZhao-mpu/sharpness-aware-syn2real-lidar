#!/bin/bash
# GPU1 follow-on: KITTI TTA batch over the full TTA list (~11 min/model x ~10 models),
# using the idle window while sampolar010_s1 trains on GPU0. Milestone lines are appended
# to lane_gpu1.out so the existing session Monitor picks them up.
set -u
PY=/home/zyf/anaconda3/envs/pointcept/bin/python
CODE=/home/zyf/paper/Syn2Real-LiDAR-DG/code
cd "$CODE" || exit 1
$PY run_evals.py --gpu 1 --only tta --targets kitti >> "$CODE/lane_gpu1_evals.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "[$(date +%F\ %H:%M:%S)] DONE kitti-tta-batch" >> "$CODE/lane_gpu1.out"
else
  echo "[$(date +%F\ %H:%M:%S)] EVAL FAIL kitti-tta-batch rc=$rc" >> "$CODE/lane_gpu1.out"
fi
echo "[$(date +%F\ %H:%M:%S)] LANE_GPU1B COMPLETE" >> "$CODE/lane_gpu1.out"
