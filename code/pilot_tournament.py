"""Fast 2-GPU pilot tournament to find which cheap mechanism reliably beats
source-only on SynLiDAR->{KITTI,STF}. Short configs (stride 20 ~10k, 12 ep).
Skips configs already finished (history.json reaching epoch>=11)."""
import os, json, time, subprocess

PY = "/home/zyf/anaconda3/envs/pointcept/bin/python"
CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
COMMON = ["--stride", "20", "--batch_size", "4", "--epochs", "12", "--lr", "0.1",
          "--warmup_iters", "300", "--eval_every", "4", "--eval_max_scans", "500",
          "--seed", "1", "--cr", "0.5"]

# (name, extra args)
CFG = [
    ("pilot_none",     ["--mode", "none", "--lambda_w", "0.0"]),                 # baseline (done)
    ("pilot_sensing",  ["--mode", "sensing", "--lambda_w", "0.1"]),              # SSW0.1 (done)
    ("pilot_dr",       ["--mode", "none", "--lambda_w", "0.0", "--dr_aug"]),     # DR-lite (running)
    ("pilot_ssw003",   ["--mode", "sensing", "--lambda_w", "0.03"]),            # SSW0.03 (running)
    ("pilot_sac05",    ["--mode", "none", "--consistency", "0.5"]),             # SAC (ours, reliable+novel)
    ("pilot_sac10",    ["--mode", "none", "--consistency", "1.0"]),
    ("pilot_sam005",   ["--mode", "none", "--sam_rho", "0.05"]),                # flat-minima
    ("pilot_sam010",   ["--mode", "none", "--sam_rho", "0.10"]),
    ("pilot_noint",    ["--mode", "none", "--no_intensity"]),                   # is intensity the culprit?
    ("pilot_dr_sac",   ["--mode", "none", "--consistency", "0.5", "--dr_aug"]), # combine aug + consistency
]


def done(name):
    h = os.path.join(EXP, name, "history.json")
    if not os.path.exists(h):
        return False
    try:
        hist = json.load(open(h))
        return bool(hist) and hist[-1]["epoch"] >= 11
    except Exception:
        return False


def running(name):
    # crude: a log file exists and process for this save_path is alive
    return any(name in l for l in os.popen("ps -eo args | grep train.py | grep -v grep").read().splitlines())


def launch(name, extra, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    logf = open(os.path.join(CODE, f"{name}.log"), "a")
    args = extra + COMMON + ["--save_path", os.path.join(EXP, name)]
    p = subprocess.Popen([PY, "train.py"] + args, cwd=CODE, env=env,
                         stdout=logf, stderr=subprocess.STDOUT)
    return p, logf


def main():
    pending = [(n, e) for n, e in CFG if not done(n) and not running(n)]
    print("pending:", [n for n, _ in pending], flush=True)
    run = {}
    while pending or run:
        for g in (0, 1):
            if g not in run and pending:
                n, e = pending.pop(0)
                p, lf = launch(n, e, g)
                run[g] = (p, n, lf)
                print(f"[{time.strftime('%H:%M:%S')}] launch {n} GPU{g}", flush=True)
                time.sleep(8)
        for g in list(run):
            p, n, lf = run[g]
            if p.poll() is not None:
                lf.close()
                print(f"[{time.strftime('%H:%M:%S')}] done {n} GPU{g} rc={p.returncode} ok={done(n)}", flush=True)
                del run[g]
        time.sleep(15)
    print("TOURNAMENT DONE", flush=True)


if __name__ == "__main__":
    main()
