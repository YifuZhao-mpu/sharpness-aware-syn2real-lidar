#!/bin/bash
# GPU1 lane: paper-critical evals FIRST (decisive headline TTA), then fill with single-route trainings.
# NOTE on CUDA_VISIBLE_DEVICES: run_evals.py self-pins its child procs via --gpu (sets CVD internally),
# so it must run with CVD UNSET (otherwise the child re-masking nests and sees no GPU). The bare
# eval.py/train.py calls do NOT self-pin (they use cuda:0), so each is prefixed with CVD=1.
set -u
PY=/home/zyf/anaconda3/envs/pointcept/bin/python
CODE=/home/zyf/paper/Syn2Real-LiDAR-DG/code
cd "$CODE" || exit 1
CR=0.5
BASE="--mode none --stride 10 --warmup_iters 500 --eval_every 12 --eval_max_scans 600 --num_points 80000 --lr 0.1 --epochs 48 --cr $CR --batch_size 4"

echo "[$(date +%F\ %H:%M:%S)] LANE_GPU1 START"

# 1) DECISIVE: TTA-STF on headline sampolar s1/s2/s3 (+ rest of TTA list). ~4 min/model, idempotent.
echo "[$(date +%F\ %H:%M:%S)] TTA-STF batch"
$PY run_evals.py --gpu 1 --only tta --targets stf >> "$CODE/lane_gpu1_evals.log" 2>&1

# 2) Complete eval for the two CKPT-ONLY pivot models (training finished, eval cut off by the GPU crash).
for m in sampolarw_s1 polarw_s1; do
  if [ ! -f "$CODE/exp/$m/eval_all.json" ] && [ -f "$CODE/exp/$m/model.pth" ]; then
    echo "[$(date +%F\ %H:%M:%S)] EVAL $m"
    CUDA_VISIBLE_DEVICES=1 $PY eval.py --ckpt "$CODE/exp/$m/model.pth" --cr $CR >> "$CODE/$m.log" 2>&1
  fi
done

# 3) Sharpness batch on existing models (skips missing/done; newly-trained models get it in the finalize pass).
echo "[$(date +%F\ %H:%M:%S)] SHARPNESS batch"
$PY run_evals.py --gpu 1 --only sharp >> "$CODE/lane_gpu1_evals.log" 2>&1

# 4) Two single-route trainings (sharpness-orthogonality inputs: aug routes are sharp, off the flat-minima line).
train_eval () {
  name="$1"; shift; extra="$*"; out="$CODE/exp/$name"
  if [ -f "$out/eval_all.json" ]; then echo "[$(date +%F\ %H:%M:%S)] SKIP $name (done)"; return 0; fi
  echo "[$(date +%F\ %H:%M:%S)] TRAIN $name :: $extra"
  CUDA_VISIBLE_DEVICES=1 $PY train.py $extra $BASE --save_path "$out" >> "$CODE/$name.log" 2>&1 || { echo "[$(date +%F\ %H:%M:%S)] TRAIN FAIL $name"; return 1; }
  CUDA_VISIBLE_DEVICES=1 $PY eval.py --ckpt "$out/model.pth" --cr $CR >> "$CODE/$name.log" 2>&1 || { echo "[$(date +%F\ %H:%M:%S)] EVAL FAIL $name"; return 1; }
  echo "[$(date +%F\ %H:%M:%S)] DONE $name"
}
train_eval weather_s1  --weather  --seed 1
train_eval lasermix_s1 --lasermix --seed 1
echo "[$(date +%F\ %H:%M:%S)] LANE_GPU1 COMPLETE"
