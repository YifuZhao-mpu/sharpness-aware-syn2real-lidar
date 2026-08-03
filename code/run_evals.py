"""Batch post-training evals: TTA (8 dihedral views) + filter-norm sharpness.
Idempotent per (model,target) and GPU-aware-by-arg. Run on existing checkpoints any time;
re-run after run_phase4.py training to pick up new models (skips finished work).
    python run_evals.py --gpu 0 --only tta --targets stf     # fast decisive STF margin
    python run_evals.py --gpu 0 --only tta --targets kitti    # slow (4071 scans x8)
    python run_evals.py --gpu 1 --only sharp                  # sharpness batch
Outputs land next to each model.pth: tta_{kitti,stf}.json, sharpness.json.
Aggregate sharpness with phase4_results.py; TTA summary printed here."""
import os, sys, json, time, argparse, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
CR = "0.5"

# TTA applied "identically to all models" (paper framing): headline + controlled baseline
# get all seeds (-> mean + TTA margin); one seed each of source-only/PolarMix/SAM shows the
# enhancement helps every route, so the SAM+PolarMix margin survives TTA.
TTA_LIST = ["sampolar_s1", "sampolar_s2", "sampolar_s3",
            "wmix_s1", "wmix_s2", "wmix_s3",
            "none_s2", "polarmix_s1", "sam005_s2", "samwmix_s1"]

# Sharpness for the Phase-4 / augmentation-route models still missing it (orthogonality
# analysis: augmentation routes are SHARP, off the flat-minima line).
SHARP_LIST = ["wmix_s1", "wmix_s2", "wmix_s3", "samwmix_s1", "samwmix_s2",
              "sampolarlm_s1", "weather_s1", "lasermix_s1",
              "sampolar010_s1", "sampolar010_s2", "pointdr_s1", "pointdr_s3"]


def ckpt(name):
    return os.path.join(EXP, name, "model.pth")


def run(cmd, gpu, logname):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, logname), "a")
    print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu} $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] -> rc={rc}", flush=True)
    return rc


def do_tta(gpu, targets):
    for n in TTA_LIST:
        if not os.path.exists(ckpt(n)):
            print(f"  skip TTA {n}: no model.pth", flush=True); continue
        missing = [t for t in targets
                   if not os.path.exists(os.path.join(EXP, n, f"tta_{t}.json"))]
        if not missing:
            print(f"  skip TTA {n}: {targets} done", flush=True); continue
        run([PY, "eval_tta.py", "--ckpt", ckpt(n), "--targets", *missing,
             "--cr", CR, "--n_rot", "4", "--max_scans", "0"], gpu, f"tta_{n}.log")


def do_sharp(gpu):
    for n in SHARP_LIST:
        if not os.path.exists(ckpt(n)):
            print(f"  skip sharp {n}: no model.pth", flush=True); continue
        if os.path.exists(os.path.join(EXP, n, "sharpness.json")):
            print(f"  skip sharp {n}: done", flush=True); continue
        run([PY, "sharpness.py", "--ckpt", ckpt(n), "--cr", CR, "--rho", "0.05"],
            gpu, f"sharp_{n}.log")


def summary():
    print("\n=== TTA summary (train-time mIoU -> +TTA mIoU) ===", flush=True)
    for n in TTA_LIST:
        e = os.path.join(EXP, n, "eval_all.json")
        base = json.load(open(e)) if os.path.exists(e) else {}
        parts = [f"  {n:16s}"]
        for t in ("kitti", "stf"):
            tf = os.path.join(EXP, n, f"tta_{t}.json")
            if not os.path.exists(tf):
                continue
            b = base.get(t, {}).get("mIoU", 0) * 100
            tt = json.load(open(tf)).get("mIoU", 0) * 100
            parts.append(f"{t.upper()} {b:5.2f}->{tt:5.2f} ({tt-b:+.2f})")
        if len(parts) > 1:
            print("   ".join(parts), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", choices=["tta", "sharp"], default=None)
    ap.add_argument("--targets", nargs="+", default=["kitti", "stf"])
    args = ap.parse_args()
    if args.only != "sharp":
        do_tta(args.gpu, args.targets)
    if args.only != "tta":
        do_sharp(args.gpu)
    summary()
    print("EVALS DONE", flush=True)


if __name__ == "__main__":
    main()
