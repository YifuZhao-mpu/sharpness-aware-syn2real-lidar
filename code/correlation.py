"""Quantify the sharpness <-> generalization relationship across ALL checkpoints that
have both sharpness.json and eval_all.json. Reports Pearson & Spearman of
(sharpness, mean real mIoU). Writes corr_stats.json for the paper."""
import os, json, glob
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    # input-augmentation methods train on a DIFFERENT distribution -> orthogonal route,
    # excluded from the same-data sharpness-vs-generalization correlation (reported separately)
    AUG = ("dr_full", "samdr", "polarmix", "sampolar",
           "polarlm", "sampolarlm", "wmix", "samwmix", "lasermix", "weather")
    rows, aug_rows = [], []
    for d in sorted(glob.glob(os.path.join(EXP, "*"))):
        n = os.path.basename(d)
        if n.startswith("pilot_") or n.startswith("_smoke") or n.startswith("cr1_"):
            continue  # cr1_* is a different backbone (kept separate for same-backbone correlation)
        if any(n.startswith(a) for a in AUG):
            sp = os.path.join(d, "sharpness.json"); ep = os.path.join(d, "eval_all.json")
            if os.path.exists(sp) and os.path.exists(ep):
                e = json.load(open(ep))
                aug_rows.append((n, json.load(open(sp))["sharpness_mean"],
                                 (e["kitti"]["mIoU"] + e["stf"]["mIoU"]) / 2 * 100))
            continue
        sp = os.path.join(d, "sharpness.json"); ep = os.path.join(d, "eval_all.json")
        if not (os.path.exists(sp) and os.path.exists(ep)):
            continue
        sh = json.load(open(sp))["sharpness_mean"]
        e = json.load(open(ep))
        m = (e["kitti"]["mIoU"] + e["stf"]["mIoU"]) / 2 * 100
        rows.append((n, sh, m, e["kitti"]["mIoU"] * 100, e["stf"]["mIoU"] * 100))
    if len(rows) < 3:
        print("not enough checkpoints with both files:", len(rows)); return
    sh = np.array([r[1] for r in rows]); mm = np.array([r[2] for r in rows])
    res = {
        "n": len(rows),
        "pearson_sharp_vs_meanmIoU": float(np.corrcoef(sh, mm)[0, 1]),
        "spearman_sharp_vs_meanmIoU": spearman(sh, mm),
        "pearson_logsharp_vs_meanmIoU": float(np.corrcoef(np.log(sh + 1e-6), mm)[0, 1]),
        "checkpoints": [{"name": r[0], "sharpness": r[1], "mean_mIoU": r[2],
                         "kitti": r[3], "stf": r[4]} for r in rows],
    }
    res["aug_methods"] = [{"name": n, "sharpness": s, "mean_mIoU": m} for n, s, m in aug_rows]
    json.dump(res, open(os.path.join(EXP, "corr_stats.json"), "w"), indent=2)
    print(f"=== SAME-DATA methods (n={res['n']}): the sharpness->generalization correlation ===")
    print(f"Pearson(sharpness, meanmIoU)     = {res['pearson_sharp_vs_meanmIoU']:.3f}")
    print(f"Spearman(sharpness, meanmIoU)    = {res['spearman_sharp_vs_meanmIoU']:.3f}")
    print(f"Pearson(log sharpness, meanmIoU) = {res['pearson_logsharp_vs_meanmIoU']:.3f}")
    for r in sorted(rows, key=lambda x: x[1]):
        print(f"  {r[0]:16s} sharp={r[1]:.4f} meanmIoU={r[2]:.2f} (K={r[3]:.1f} S={r[4]:.1f})")
    print(f"=== INPUT-AUGMENTATION methods (separate axis, n={len(aug_rows)}) ===")
    for n, s, m in sorted(aug_rows, key=lambda x: x[1]):
        print(f"  {n:16s} sharp={s:.4f} meanmIoU={m:.2f}")


if __name__ == "__main__":
    main()
