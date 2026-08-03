"""Target-free model selection: which SOURCE-computable signal best ranks models by TARGET
mIoU? Compares sharpness vs source loss vs source mIoU (all computable without target data)
across the same-data checkpoints. Reports Spearman rank-correlation of each signal with mean
target mIoU, plus the 'select the best by this signal' regret. Writes selection_stats.json."""
import os, json, glob
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
AUG = ("dr_full", "samdr", "polarmix", "sampolar")


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(EXP, "*"))):
        n = os.path.basename(d)
        if n.startswith(("pilot_", "_smoke", "cr1_")) or any(n.startswith(a) for a in AUG):
            continue
        sp = os.path.join(d, "sharpness.json"); ep = os.path.join(d, "eval_all.json")
        sm = os.path.join(d, "source_metrics.json")
        if not (os.path.exists(sp) and os.path.exists(ep) and os.path.exists(sm)):
            continue
        e = json.load(open(ep)); s = json.load(open(sm))
        tgt = (e["kitti"]["mIoU"] + e["stf"]["mIoU"]) / 2 * 100
        rows.append({"name": n, "sharpness": json.load(open(sp))["sharpness_mean"],
                     "src_loss": s["src_loss"], "src_mIoU": s["src_mIoU"], "target": tgt})
    if len(rows) < 4:
        print("not enough checkpoints with all 3 files:", len(rows)); return
    tgt = np.array([r["target"] for r in rows])
    # selection signals (sign chosen so that LOWER signal -> predict HIGHER target where applicable)
    signals = {
        "sharpness (lower=better)":   (-np.array([r["sharpness"] for r in rows])),
        "source mIoU (higher=better)": np.array([r["src_mIoU"] for r in rows]),
        "-source loss (lower=better)": (-np.array([r["src_loss"] for r in rows])),
    }
    print(f"n={len(rows)} same-data checkpoints. Spearman(signal, target mIoU):")
    res = {"n": len(rows), "spearman": {}, "pick_regret": {}}
    best_target = tgt.max()
    for name, sig in signals.items():
        rho = spearman(sig, tgt)
        pick = int(np.argmax(sig))                      # model this signal would select
        regret = best_target - tgt[pick]                # how far from the true best
        res["spearman"][name] = rho
        res["pick_regret"][name] = {"picked": rows[pick]["name"], "regret_mIoU": float(regret)}
        print(f"  {name:30s} Spearman={rho:+.3f}  picks {rows[pick]['name']:14s} (regret {regret:.2f} mIoU)")
    json.dump({"rows": rows, **res}, open(os.path.join(EXP, "selection_stats.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
