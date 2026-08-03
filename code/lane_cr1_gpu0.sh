#!/bin/bash
# cr=1.0 retrain lane, GPU0: cr1_sam_s2 (SAM rho=0.05, seed 2, full-width MinkUNet, ~17h @ ~1284s/ep).
# Protocol replicated from exp/cr1_sam_s1/args.json: bs3, cr1.0, stride10, 48ep, lr0.1.
# train -> eval(--cr 1.0) -> sharpness (paper probe: 5 dirs x 6 batches). Idempotent via eval_all.json.
set -u
PY=/home/zyf/anaconda3/envs/pointcept/bin/python
CODE=/home/zyf/paper/Syn2Real-LiDAR-DG/code
cd "$CODE" || exit 1
export CUDA_VISIBLE_DEVICES=0
BASE="--mode none --stride 10 --warmup_iters 500 --eval_every 12 --eval_max_scans 600 --num_points 80000 --lr 0.1 --epochs 48 --cr 1.0 --batch_size 3"

name=cr1_sam_s2; out="$CODE/exp/$name"
echo "[$(date +%F\ %H:%M:%S)] LANE_CR1_GPU0 START"
if [ -f "$out/eval_all.json" ]; then
  echo "[$(date +%F\ %H:%M:%S)] SKIP $name (done)"
else
  rm -f "$out/model.pth" "$out/history.json"   # stale partial (killed at ep22) — retrain from scratch
  echo "[$(date +%F\ %H:%M:%S)] TRAIN $name :: --sam_rho 0.05 --seed 2 (cr=1.0 bs3)"
  $PY train.py --sam_rho 0.05 --seed 2 $BASE --save_path "$out" >> "$CODE/$name.log" 2>&1 \
    || { echo "[$(date +%F\ %H:%M:%S)] TRAIN FAIL $name"; exit 1; }
  echo "[$(date +%F\ %H:%M:%S)] EVAL $name"
  $PY eval.py --ckpt "$out/model.pth" --cr 1.0 >> "$CODE/$name.log" 2>&1 \
    || { echo "[$(date +%F\ %H:%M:%S)] EVAL FAIL $name"; exit 1; }
  $PY sharpness.py --ckpt "$out/model.pth" --cr 1.0 --rho 0.05 --n_dirs 5 --n_batches 6 >> "$CODE/$name.log" 2>&1 \
    || echo "[$(date +%F\ %H:%M:%S)] SHARP FAIL $name (non-fatal)"
  echo "[$(date +%F\ %H:%M:%S)] DONE $name"
fi
echo "[$(date +%F\ %H:%M:%S)] LANE_CR1_GPU0 COMPLETE"
