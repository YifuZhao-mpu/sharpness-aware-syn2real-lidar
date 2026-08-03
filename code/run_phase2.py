"""Phase-2 experiments addressing reviewer round-1 (priority order):
1. Compute-matched control: source-only at 2x epochs (rules out 'SAM = 2x compute').
2. PolarMix strong protocol-matched DG baseline (+ SAM on top).
3. cr=1.0 backbone generality (source-only vs SAM).
Each job = train + full eval. Skips done. Usage: python run_phase2.py 0 1"""
import os, sys, time, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
BASE = ["--stride", "10", "--warmup_iters", "500", "--eval_every", "12",
        "--eval_max_scans", "600", "--num_points", "80000", "--lr", "0.1"]

def J(name, extra, cr="0.5"):
    return (name, extra + ["--cr", cr], cr)

CFG = [
    # cr1_none_s1 + control running (orphaned). Priority next: combination seed2 + cr1 SAM (generality)
    J("sampolar_s1",  ["--mode", "none", "--sam_rho", "0.05", "--polarmix", "--epochs", "48", "--batch_size", "4", "--seed", "1"]),  # done
    J("none_long_s1", ["--mode", "none", "--epochs", "96", "--batch_size", "4", "--seed", "1", "--grad_clip", "10.0"]),  # running
    J("cr1_none_s1",  ["--mode", "none", "--epochs", "48", "--batch_size", "3", "--seed", "1"], cr="1.0"),  # running
    J("sampolar_s2",  ["--mode", "none", "--sam_rho", "0.05", "--polarmix", "--epochs", "48", "--batch_size", "4", "--seed", "2"]),  # combination seed2 (prioritized)
    J("cr1_sam_s1",   ["--mode", "none", "--sam_rho", "0.05", "--epochs", "48", "--batch_size", "3", "--seed", "1"], cr="1.0"),  # cr1 SAM for generality
    J("sampolar_s3",  ["--mode", "none", "--sam_rho", "0.05", "--polarmix", "--epochs", "48", "--batch_size", "4", "--seed", "3"]),  # combination seed3
    J("none_long_s2", ["--mode", "none", "--epochs", "96", "--batch_size", "4", "--seed", "2", "--grad_clip", "10.0"]),  # control seed2
    J("cr1_none_s2",  ["--mode", "none", "--epochs", "48", "--batch_size", "3", "--seed", "2"], cr="1.0"),
    J("cr1_sam_s2",   ["--mode", "none", "--sam_rho", "0.05", "--epochs", "48", "--batch_size", "3", "--seed", "2"], cr="1.0"),
]


def done(name):
    return os.path.exists(os.path.join(EXP, name, "eval_all.json"))


def running(name):
    return any((("save_path " + os.path.join(EXP, name)) in l)
               for l in os.popen("ps -eo args | grep train.py | grep -v grep").read().splitlines())


def gpu_free(g, thresh_mib=2500):
    """True if GPU g is idle (low memory use) -> safe to launch, even alongside external jobs."""
    try:
        out = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {g}").read()
        return int(out.strip().split("\n")[0]) < thresh_mib
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
            if g not in run and pending and gpu_free(g):   # only launch on an idle GPU
                n, e, cr = pending.pop(0)
                if done(n) or running(n):
                    continue
                p, lf = launch(n, e, cr, g)
                run[g] = (p, n, lf)
                print(f"[{time.strftime('%H:%M:%S')}] launch {n} GPU{g}", flush=True)
                time.sleep(20)  # let memory ramp before re-checking gpu_free
        for g in list(run):
            p, n, lf = run[g]
            if p.poll() is not None:
                lf.close()
                print(f"[{time.strftime('%H:%M:%S')}] DONE {n} GPU{g} rc={p.returncode} ok={done(n)}", flush=True)
                del run[g]
        time.sleep(20)
    print("PHASE2 DONE", flush=True)


if __name__ == "__main__":
    main()
