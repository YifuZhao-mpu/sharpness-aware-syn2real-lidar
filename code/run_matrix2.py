"""SAM-centered full experiment matrix. Single-GPU jobs, configurable GPU set.
Each job = train.py (48 ep, ~19k) then full eval.py. Skips jobs with eval_all.json.
Honest DG protocol: report FINAL-epoch model (no target-based selection).
Usage:  python run_matrix2.py 0        # use GPU0 only
        python run_matrix2.py 0 1      # use both GPUs
"""
import os, sys, time, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
COMMON = ["--stride", "10", "--batch_size", "4", "--epochs", "48", "--lr", "0.1",
          "--warmup_iters", "500", "--eval_every", "8", "--eval_max_scans", "600",
          "--cr", "0.5", "--num_points", "80000"]

def J(name, extra, seed):
    return (name, extra + ["--seed", str(seed)])

CFG = [
    # main comparison (3 seeds): source-only vs SAM(rho=0.05)
    J("none_full",   ["--mode", "none"], 1),                 # DONE
    J("none_s2",     ["--mode", "none"], 2),
    J("none_s3",     ["--mode", "none"], 3),
    J("sam005_full", ["--mode", "none", "--sam_rho", "0.05"], 1),  # running
    J("sam005_s2",   ["--mode", "none", "--sam_rho", "0.05"], 2),
    J("sam005_s3",   ["--mode", "none", "--sam_rho", "0.05"], 3),
    # novelty: sensing-aware SAM (prioritized) — sharpness ascent on a sim->real sensing-perturbed view
    J("samsens_s1",  ["--mode", "none", "--sam_rho", "0.05", "--sam_sensing"], 1),
    J("samsens_s2",  ["--mode", "none", "--sam_rho", "0.05", "--sam_sensing"], 2),
    # rho sweep (1 seed)
    J("sam010_s1",   ["--mode", "none", "--sam_rho", "0.10"], 1),
    J("sam001_s1",   ["--mode", "none", "--sam_rho", "0.01"], 1),
    J("sam020_s1",   ["--mode", "none", "--sam_rho", "0.20"], 1),
    # baselines / complementarity (1 seed)
    J("dr_full_s1",  ["--mode", "none", "--dr_aug"], 1),
    J("samdr_s1",    ["--mode", "none", "--sam_rho", "0.05", "--dr_aug"], 1),
    # SWA: second flat-minima method (preempts "why not SWA?"; tests the flat-minima thesis itself)
    J("swa_s1",      ["--mode", "none", "--swa_start", "36"], 1),
    J("swa_s2",      ["--mode", "none", "--swa_start", "36"], 2),
    J("swa_s3",      ["--mode", "none", "--swa_start", "36"], 3),
]


def done(name):
    return os.path.exists(os.path.join(EXP, name, "eval_all.json"))


def running(name):
    return any((os.path.join("exp", name) in l) or (("save_path " + os.path.join(EXP, name)) in l)
               for l in os.popen("ps -eo args | grep train.py | grep -v grep").read().splitlines())


def launch(name, extra, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, f"{name}.log"), "a")
    train = [PY, "train.py"] + extra + COMMON + ["--save_path", os.path.join(EXP, name)]
    ev = [PY, "eval.py", "--ckpt", os.path.join(EXP, name, "model.pth"), "--cr", "0.5"]
    cmd = " ".join(train) + " && " + " ".join(ev)
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def main():
    gpus = [int(x) for x in sys.argv[1:]] or [0]
    pending = [(n, e) for n, e in CFG if not done(n) and not running(n)]
    print(f"GPUs={gpus}  pending={[n for n,_ in pending]}", flush=True)
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
    print("MATRIX2 DONE", flush=True)


if __name__ == "__main__":
    main()
