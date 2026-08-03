"""Unattended 2-GPU experiment orchestrator. Runs jobs two-at-a-time (one per GPU),
each job = train.py then full eval.py. Resumable: skips jobs whose eval_all.json exists.
Edit HP constants below after the pilot validates signal."""
import os, sys, json, time, subprocess, itertools

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")

# ---- hyperparameters (tune after pilot) ----
EPOCHS = 35
EPOCHS_ABL = 30
STRIDE = 10          # ~19k SynLiDAR scans (PointDR-comparable)
NUMPTS = 80000
BATCH = 6
LR = 0.12
WARM = 500
EVAL_EVERY = 10
EVAL_MAX = 400       # subset eval during training; full eval done post-hoc
CR = 0.5
LAM = 0.1            # best lambda from pilot/sweep

def job(name, mode, seed, lam=0.0, dr=False, noint=False, epochs=EPOCHS):
    args = ["--mode", mode, "--lambda_w", str(lam), "--epochs", str(epochs),
            "--batch_size", str(BATCH), "--lr", str(LR), "--warmup_iters", str(WARM),
            "--stride", str(STRIDE), "--num_points", str(NUMPTS), "--cr", str(CR),
            "--eval_every", str(EVAL_EVERY), "--eval_max_scans", str(EVAL_MAX),
            "--seed", str(seed), "--save_path", os.path.join(EXP, name)]
    if dr: args.append("--dr_aug")
    if noint: args.append("--no_intensity")
    return {"name": name, "args": args, "noint": noint}

JOBS = []
# main
for s in (1, 2, 3):
    JOBS.append(job(f"none_s{s}", "none", s, 0.0))
    JOBS.append(job(f"ssw_s{s}", "sensing", s, LAM))
JOBS.append(job("blunt_s1", "blunt", 1, LAM))
JOBS.append(job("dr_s1", "none", 1, 0.0, dr=True))            # DR-lite public-style DG baseline
JOBS.append(job("ssw_dr_s1", "sensing", 1, LAM, dr=True))     # complementarity: SSW on top of DR
# ablations (single seed)
JOBS.append(job("random_s1", "random", 1, LAM, epochs=EPOCHS_ABL))
for lam in (0.05, 0.2, 0.5):
    JOBS.append(job(f"ssw_lam{lam}_s1", "sensing", 1, lam, epochs=EPOCHS_ABL))
JOBS.append(job("none_noint_s1", "none", 1, 0.0, noint=True, epochs=EPOCHS_ABL))
JOBS.append(job("ssw_noint_s1", "sensing", 1, LAM, noint=True, epochs=EPOCHS_ABL))


def done(j):
    return os.path.exists(os.path.join(EXP, j["name"], "eval_all.json"))


def launch(j, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, f"log_{j['name']}.log"), "w")
    # chain: train then full eval
    ckpt = os.path.join(EXP, j["name"], "model.pth")
    eval_args = ["--ckpt", ckpt, "--cr", str(CR)] + (["--no_intensity"] if j["noint"] else [])
    cmd = (f"{PY} train.py " + " ".join(j['args']) +
           f" && {PY} eval.py " + " ".join(eval_args))
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def main():
    gpus = [0, 1]
    pending = [j for j in JOBS if not done(j)]
    print(f"{len(JOBS)} jobs, {len(pending)} pending", flush=True)
    running = {}  # gpu -> (proc, job, logf)
    while pending or running:
        for g in gpus:
            if g not in running and pending:
                j = pending.pop(0)
                if done(j):
                    continue
                p, lf = launch(j, g)
                running[g] = (p, j, lf)
                print(f"[{time.strftime('%H:%M:%S')}] launch {j['name']} on GPU{g}", flush=True)
        for g in list(running):
            p, j, lf = running[g]
            if p.poll() is not None:
                lf.close()
                ok = done(j)
                print(f"[{time.strftime('%H:%M:%S')}] finish {j['name']} GPU{g} rc={p.returncode} eval_ok={ok}", flush=True)
                del running[g]
        time.sleep(15)
    print("ALL JOBS DONE", flush=True)


if __name__ == "__main__":
    main()
