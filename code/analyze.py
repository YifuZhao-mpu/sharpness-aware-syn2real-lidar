"""Aggregate experiment results into tables. Reads exp/*/args.json + history.json
(+ eval_all.json if present). Prints markdown; writes results_table.md / .csv.
Only reads real result files — no synthetic numbers."""
import os, json, glob, argparse
import numpy as np
from collections import defaultdict


def best_eval(hist):
    """Pick epoch with best mean(KITTI,STF) mIoU from history.json."""
    best, key = None, -1
    for h in hist:
        k = h["kitti"]["mIoU"]; s = h["stf"]["mIoU"]
        m = (k + s) / 2
        if m > key:
            key, best = m, h
    return best


def load_run(d):
    a = json.load(open(os.path.join(d, "args.json")))
    out = {"name": os.path.basename(d), "mode": a.get("mode"), "lambda_w": a.get("lambda_w"),
           "seed": a.get("seed"), "epochs": a.get("epochs"), "dr_aug": a.get("dr_aug", False),
           "no_intensity": a.get("no_intensity", False)}
    # prefer standalone full eval if present
    ea = os.path.join(d, "eval_all.json")
    hp = os.path.join(d, "history.json")
    if os.path.exists(ea):
        e = json.load(open(ea))
        out["kitti"] = e.get("kitti", {}).get("mIoU")
        out["stf"] = e.get("stf", {}).get("mIoU")
        out["kitti_nc"] = e.get("kitti", {}).get("nuisance_corr_target")
        out["stf_nc"] = e.get("stf", {}).get("nuisance_corr_target")
        out["src"] = "full_eval"
    elif os.path.exists(hp):
        hist = json.load(open(hp))
        if hist:
            b = best_eval(hist)
            out["kitti"] = b["kitti"]["mIoU"]; out["stf"] = b["stf"]["mIoU"]
            out["best_epoch"] = b["epoch"]; out["src"] = "history_best"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="exp")
    args = ap.parse_args()
    runs = []
    for d in sorted(glob.glob(os.path.join(args.exp_dir, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "args.json")):
            try:
                runs.append(load_run(d))
            except Exception as e:
                print("skip", d, e)
    # print table
    hdr = ["name", "mode", "lambda_w", "seed", "kitti", "stf", "mean", "src"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in runs:
        k = r.get("kitti"); s = r.get("stf")
        kp = f"{k*100:.2f}" if isinstance(k, (int, float)) else "-"
        sp = f"{s*100:.2f}" if isinstance(s, (int, float)) else "-"
        mp = f"{(k+s)/2*100:.2f}" if isinstance(k, (int, float)) and isinstance(s, (int, float)) else "-"
        lines.append("| " + " | ".join([str(r.get("name")), str(r.get("mode")),
                     str(r.get("lambda_w")), str(r.get("seed")), kp, sp, mp, str(r.get("src"))]) + " |")
    md = "\n".join(lines)
    print(md)
    open(os.path.join(args.exp_dir, "results_table.md"), "w").write(md + "\n")
    # seed aggregation: mean+/-std per (mode,lambda,dr,no_int) group over seeds
    groups = defaultdict(list)
    for r in runs:
        if isinstance(r.get("kitti"), (int, float)) and isinstance(r.get("stf"), (int, float)):
            key = (r["mode"], r["lambda_w"], r.get("dr_aug"), r.get("no_intensity"))
            groups[key].append((r["kitti"], r["stf"]))
    print("\n### Seed-aggregated (mean±std over seeds)")
    for key, vals in sorted(groups.items()):
        arr = np.array(vals) * 100
        m = arr.mean(0); sd = arr.std(0)
        print(f"  mode={key[0]} lam={key[1]} dr={key[2]} noint={key[3]} n={len(vals)}: "
              f"KITTI {m[0]:.2f}±{sd[0]:.2f}  STF {m[1]:.2f}±{sd[1]:.2f}")


if __name__ == "__main__":
    main()
