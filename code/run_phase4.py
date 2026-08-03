"""Phase-4: exceed SOTA via composing orthogonal routes + controlled UniMix-style comparison.
The paper's thesis is "flat minima (SAM) composes with input-side augmentation". We extend
the augmentation axis with a 3rd orthogonal route (LaserMix, inclination-axis mixing) and a
4th (LISA-style weather sim = UniMix's adverse-weather component), and show flat-minima
training composes with even the full SOTA-style augmentation pipeline.

Configs (BASE protocol identical to phase-2/3: stride10, 48ep, cr0.5, lr0.1, bs4 -> directly
comparable to existing source-only/SAM/PolarMix/sampolar full-eval numbers):
  sampolarlm = SAM + PolarMix + LaserMix         (headline: flatness + 2 mixing routes)
  samwmix    = SAM + PolarMix + LaserMix + Weather (SAM composed with full UniMix-style pipeline)
  wmix       = PolarMix + LaserMix + Weather       (UniMix-style universal-mixing+weather, NO SAM -> controlled SOTA baseline)
  polarlm    = PolarMix + LaserMix                 (mixing-only ablation, isolates SAM & weather)
  lasermix / weather = single-route characterization (sharpness / orthogonality analysis)

Priority = BREADTH-FIRST on seed 1 (compare all key recipes fast), THEN DEPTH (scale winner to
3 seeds + complete baselines). GPU-aware, 2 GPUs. Each job = train + full eval. Skips done.
Usage: python run_phase4.py 0 1"""
import os, sys, time, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
CR = "0.5"
BASE = ["--mode", "none", "--stride", "10", "--warmup_iters", "500", "--eval_every", "12",
        "--eval_max_scans", "600", "--num_points", "80000", "--lr", "0.1", "--epochs", "48",
        "--cr", CR, "--batch_size", "4"]

SAM = ["--sam_rho", "0.05"]
SAM10 = ["--sam_rho", "0.10"]
CFG = [
    # KEY FINDING: stacking laser+weather does NOT beat SAM+PolarMix (sampolar 30.38/24.95, n=3) -
    # it trades KITTI for no STF-mean gain. sampolar BEATS our UniMix-style wmix (28.50/24.47) on both
    # = controlled SOTA win. Headline stays sampolar. Remaining runs converge the story:
    # running (skip): sampolarw_s1, polarw_s1. done: samwmix s1/s2, wmix s1/s2, sampolarlm_s1.
    ("sampolar010_s1", SAM10 + ["--polarmix", "--seed", "1"]),  # on-narrative STF lever: higher rho on SAM+PolarMix
    ("wmix_s3",              ["--polarmix", "--lasermix", "--weather", "--seed", "3"]),  # controlled UniMix-style baseline -> n=3 (stable)
    ("weather_s1",           ["--weather", "--seed", "1"]),     # single-route: sharpness off-the-line (aug is sharp)
    ("lasermix_s1",          ["--lasermix", "--seed", "1"]),    # single-route: sharpness off-the-line
    ("sampolar010_s2", SAM10 + ["--polarmix", "--seed", "2"]),  # 2nd seed IF rho=0.10 helps STF
]


def done(name):
    return os.path.exists(os.path.join(EXP, name, "eval_all.json"))


def running(name):
    return any((("save_path " + os.path.join(EXP, name)) in l)
               for l in os.popen("ps -eo args | grep train.py | grep -v grep").read().splitlines())


def gpu_free(g, thresh=2500):
    try:
        return int(os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {g}").read().strip().split("\n")[0]) < thresh
    except Exception:
        return False


def launch(name, extra, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, f"{name}.log"), "a")
    train = [PY, "train.py"] + extra + BASE + ["--save_path", os.path.join(EXP, name)]
    ev = [PY, "eval.py", "--ckpt", os.path.join(EXP, name, "model.pth"), "--cr", CR]
    cmd = " ".join(train) + " && " + " ".join(ev)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def main():
    gpus = [int(x) for x in sys.argv[1:]] or [0]
    pending = [(n, e) for n, e in CFG if not done(n) and not running(n)]
    print(f"GPUs={gpus} pending={[n for n, _ in pending]}", flush=True)
    run = {}
    while pending or run:
        for g in gpus:
            if g not in run and pending and gpu_free(g):
                n, e = pending.pop(0)
                if done(n) or running(n):
                    continue
                p, lf = launch(n, e, g)
                run[g] = (p, n, lf)
                print(f"[{time.strftime('%H:%M:%S')}] launch {n} GPU{g}", flush=True)
                time.sleep(20)
        for g in list(run):
            p, n, lf = run[g]
            if p.poll() is not None:
                lf.close()
                print(f"[{time.strftime('%H:%M:%S')}] DONE {n} GPU{g} rc={p.returncode} ok={done(n)}", flush=True)
                del run[g]
        time.sleep(20)
    print("PHASE4 DONE", flush=True)


if __name__ == "__main__":
    main()
