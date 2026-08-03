"""Backbone-generality experiment: does the flat-minima effect hold on a different-capacity
backbone? Full-width MinkUNet (cr=1.0, ~39M params, 4x the main model). source-only vs SAM.
Same protocol otherwise. Each job = train + full eval. Skips done (eval_all.json).
Usage: python run_backbone.py 0 1"""
import os, sys, time, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
# cr=1.0 is ~3x heavier; keep 48 ep but smaller batch to fit memory headroom
COMMON = ["--stride", "10", "--batch_size", "3", "--epochs", "48", "--lr", "0.1",
          "--warmup_iters", "500", "--eval_every", "12", "--eval_max_scans", "600",
          "--cr", "1.0", "--num_points", "80000"]

def J(name, extra, seed):
    return (name, extra + ["--seed", str(seed)])

CFG = [
    J("cr1_none_s1", ["--mode", "none"], 1),
    J("cr1_sam_s1",  ["--mode", "none", "--sam_rho", "0.05"], 1),
    J("cr1_none_s2", ["--mode", "none"], 2),
    J("cr1_sam_s2",  ["--mode", "none", "--sam_rho", "0.05"], 2),
]


def done(name):
    return os.path.exists(os.path.join(EXP, name, "eval_all.json"))


def running(name):
    return any((("save_path " + os.path.join(EXP, name)) in l)
               for l in os.popen("ps -eo args | grep train.py | grep -v grep").read().splitlines())


def launch(name, extra, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, f"{name}.log"), "a")
    train = [PY, "train.py"] + extra + COMMON + ["--save_path", os.path.join(EXP, name)]
    ev = [PY, "eval.py", "--ckpt", os.path.join(EXP, name, "model.pth"), "--cr", "1.0"]
    cmd = " ".join(train) + " && " + " ".join(ev)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def main():
    gpus = [int(x) for x in sys.argv[1:]] or [0]
    pending = [(n, e) for n, e in CFG if not done(n) and not running(n)]
    print(f"GPUs={gpus} pending={[n for n,_ in pending]}", flush=True)
    run = {}
    while pending or run:
        for g in gpus:
            if g not in run and pending:
                n, e = pending.pop(0)
                if done(n) or running(n):
                    continue
                p, lf = launch(n, e, g)
                run[g] = (p, n, lf)
                print(f"[{time.strftime('%H:%M:%S')}] launch {n} GPU{g}", flush=True)
                time.sleep(10)
        for g in list(run):
            p, n, lf = run[g]
            if p.poll() is not None:
                lf.close()
                print(f"[{time.strftime('%H:%M:%S')}] DONE {n} GPU{g} rc={p.returncode} ok={done(n)}", flush=True)
                del run[g]
        time.sleep(20)
    print("BACKBONE DONE", flush=True)


if __name__ == "__main__":
    main()
