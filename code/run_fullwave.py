"""Decisive full-scale comparison wave (~19k SynLiDAR, 48 ep). 2 GPUs, 2-per-wave.
Each job = train.py then full eval.py (KITTI 4071 + STF 250 -> eval_all.json).
Skips jobs whose eval_all.json already exists."""
import os, time, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
COMMON = ["--stride", "10", "--batch_size", "4", "--epochs", "48", "--lr", "0.1",
          "--warmup_iters", "500", "--eval_every", "8", "--eval_max_scans", "600",
          "--seed", "1", "--cr", "0.5", "--num_points", "80000"]

CFG = [
    ("none_full",   ["--mode", "none"]),
    ("sac05_full",  ["--mode", "none", "--consistency", "0.5"]),
    ("sac10_full",  ["--mode", "none", "--consistency", "1.0"]),
    ("sam005_full", ["--mode", "none", "--sam_rho", "0.05"]),
]


def done(name):
    return os.path.exists(os.path.join(EXP, name, "eval_all.json"))


def launch(name, extra, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, f"{name}.log"), "a")
    train = [PY, "train.py"] + extra + COMMON + ["--save_path", os.path.join(EXP, name)]
    ckpt = os.path.join(EXP, name, "model.pth")
    ev = [PY, "eval.py", "--ckpt", ckpt, "--cr", "0.5"]
    cmd = " ".join(train) + " && " + " ".join(ev)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def main():
    pending = [(n, e) for n, e in CFG if not done(n)]
    print("pending:", [n for n, _ in pending], flush=True)
    run = {}
    while pending or run:
        for g in (0, 1):
            if g not in run and pending:
                n, e = pending.pop(0)
                p, lf = launch(n, e, g)
                run[g] = (p, n, lf)
                print(f"[{time.strftime('%H:%M:%S')}] launch {n} GPU{g}", flush=True)
                time.sleep(10)
        for g in list(run):
            p, n, lf = run[g]
            if p.poll() is not None:
                lf.close()
                print(f"[{time.strftime('%H:%M:%S')}] DONE {n} GPU{g} rc={p.returncode} eval_ok={done(n)}", flush=True)
                del run[g]
        time.sleep(20)
    print("FULLWAVE DONE", flush=True)


if __name__ == "__main__":
    main()
