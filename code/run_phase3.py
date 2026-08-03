"""Phase-3: controlled SOTA comparison (PointDR reimpl, 3 seeds) + cr=1.0 extra seeds for
generality robustness. GPU-aware, 2 GPUs. Each job = train + full eval. Skips done.
Usage: python run_phase3.py 0 1"""
import os, sys, time, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
BASE = ["--stride", "10", "--warmup_iters", "500", "--eval_every", "12",
        "--eval_max_scans", "600", "--num_points", "80000", "--lr", "0.1", "--epochs", "48"]

def J(name, extra, cr="0.5"):
    return (name, extra + ["--cr", cr], cr)

CFG = [
    # PointDR (Xiao CVPR'23) reimplementation under our exact protocol -- controlled SOTA baseline
    J("pointdr_s1", ["--pointdr", "--pointdr_lambda", "0.1", "--batch_size", "4", "--seed", "1"]),
    J("pointdr_s2", ["--pointdr", "--pointdr_lambda", "0.1", "--batch_size", "4", "--seed", "2"]),
    J("pointdr_s3", ["--pointdr", "--pointdr_lambda", "0.1", "--batch_size", "4", "--seed", "3"]),
    # cr=1.0 extra seeds (generality robustness -> 3 seeds each)
    J("cr1_none_s2", ["--mode", "none", "--batch_size", "3", "--seed", "2"], cr="1.0"),
    J("cr1_sam_s2",  ["--mode", "none", "--sam_rho", "0.05", "--batch_size", "3", "--seed", "2"], cr="1.0"),
    J("cr1_none_s3", ["--mode", "none", "--batch_size", "3", "--seed", "3"], cr="1.0"),
    J("cr1_sam_s3",  ["--mode", "none", "--sam_rho", "0.05", "--batch_size", "3", "--seed", "3"], cr="1.0"),
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


def launch(name, extra, cr, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, f"{name}.log"), "a")
    train = [PY, "train.py"] + extra + BASE + ["--save_path", os.path.join(EXP, name)]
    ev = [PY, "eval.py", "--ckpt", os.path.join(EXP, name, "model.pth"), "--cr", cr]
    cmd = " ".join(train) + " && " + " ".join(ev)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def main():
    gpus = [int(x) for x in sys.argv[1:]] or [0]
    pending = [(n, e, cr) for n, e, cr in CFG if not done(n) and not running(n)]
    print(f"GPUs={gpus} pending={[n for n,_,_ in pending]}", flush=True)
    run = {}
    while pending or run:
        for g in gpus:
            if g not in run and pending and gpu_free(g):
                n, e, cr = pending.pop(0)
                if done(n) or running(n):
                    continue
                p, lf = launch(n, e, cr, g)
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
    print("PHASE3 DONE", flush=True)


if __name__ == "__main__":
    main()
