#!/bin/bash
# GPU0 training lane: the two highest-value trainings.
#   wmix_s3        = PolarMix+LaserMix+Weather, seed 3 -> completes controlled UniMix-style baseline to n=3 (key comparison row)
#   sampolar010_s1 = SAM(rho=0.10)+PolarMix, seed 1   -> on-narrative STF lever (flatter minima via larger radius)
# Idempotent: skips a job whose eval_all.json already exists (resumable after interruption).
set -u
PY=/home/zyf/anaconda3/envs/pointcept/bin/python
CODE=/home/zyf/paper/Syn2Real-LiDAR-DG/code
cd "$CODE" || exit 1
export CUDA_VISIBLE_DEVICES=0
CR=0.5
BASE="--mode none --stride 10 --warmup_iters 500 --eval_every 12 --eval_max_scans 600 --num_points 80000 --lr 0.1 --epochs 48 --cr $CR --batch_size 4"

train_eval () {
  name="$1"; shift; extra="$*"; out="$CODE/exp/$name"
  if [ -f "$out/eval_all.json" ]; then echo "[$(date +%F\ %H:%M:%S)] SKIP $name (done)"; return 0; fi
  echo "[$(date +%F\ %H:%M:%S)] TRAIN $name :: $extra"
  $PY train.py $extra $BASE --save_path "$out" >> "$CODE/$name.log" 2>&1 || { echo "[$(date +%F\ %H:%M:%S)] TRAIN FAIL $name"; return 1; }
  echo "[$(date +%F\ %H:%M:%S)] EVAL $name"
  $PY eval.py --ckpt "$out/model.pth" --cr $CR >> "$CODE/$name.log" 2>&1 || { echo "[$(date +%F\ %H:%M:%S)] EVAL FAIL $name"; return 1; }
  echo "[$(date +%F\ %H:%M:%S)] DONE $name"
}

echo "[$(date +%F\ %H:%M:%S)] LANE_GPU0 START"
train_eval wmix_s3        --polarmix --lasermix --weather --seed 3
train_eval sampolar010_s1 --sam_rho 0.10 --polarmix --seed 1
echo "[$(date +%F\ %H:%M:%S)] LANE_GPU0 COMPLETE"
