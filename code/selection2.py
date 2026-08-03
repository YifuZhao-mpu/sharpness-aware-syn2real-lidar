"""Regenerated source-computable model-selection analysis (Supplementary Note 2).
Supersedes selection_stats.json (old n=14 cohort, coarse tensor probe). Uses the
rigorous probe (sharpness2.json) over the frozen n=13 cohort restricted to the 12
checkpoints with recorded held-out source metrics (none_long_s1 lacks them).
Writes exp/selection_stats2.json."""
import os, json
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")

old = json.load(open(os.path.join(EXP, "selection_stats.json")))
src = {r["name"]: (r["src_loss"], r["src_mIoU"]) for r in old["rows"]}

COHORT = ["none_full", "none_s2", "none_s3", "sam001_s1", "sam005_full",
          "sam005_s2", "sam005_s3", "sam010_s1", "sam020_s1",
          "swa_s1", "swa_s2", "swa_s3"]  # frozen cohort minus none_long_s1 (no src metrics)

rows = []
for n in COHORT:
    s2 = json.load(open(os.path.join(EXP, n, "sharpness2.json")))["sharpness_mean"]
    e = json.load(open(os.path.join(EXP, n, "eval_all.json")))
    tgt = (e["kitti"]["mIoU"] + e["stf"]["mIoU"]) / 2 * 100
    rows.append({"name": n, "sharpness2": s2, "src_loss": src[n][0],
                 "src_mIoU": src[n][1], "target": tgt})

best = max(rows, key=lambda r: r["target"])
picks = {"sharpness2 (min)": min(rows, key=lambda r: r["sharpness2"]),
         "src_loss (min)": min(rows, key=lambda r: r["src_loss"]),
         "src_mIoU (max)": max(rows, key=lambda r: r["src_mIoU"])}
res = {"note": "supersedes selection_stats.json (old cohort + coarse probe)",
       "cohort": COHORT, "excluded": {"none_long_s1": "no held-out source metrics recorded"},
       "true_best": {"name": best["name"], "target": best["target"]},
       "selections": {k: {"name": v["name"], "target": v["target"],
                          "regret": round(best["target"] - v["target"], 4)}
                      for k, v in picks.items()},
       "rows": rows}
json.dump(res, open(os.path.join(EXP, "selection_stats2.json"), "w"), indent=2)
print("true best:", best["name"], round(best["target"], 2))
for k, v in res["selections"].items():
    print(f"{k:18s} -> {v['name']:14s} regret {v['regret']}")
