"""Honest statistics for the Scientific Reports rewrite (round-9 P7 response).

Recomputes every headline comparison from the raw eval_all.json files with:
  - sample SD (ddof=1), not population SD
  - seed-paired differences with two-sided paired t-tests and 95% CIs
  - all per-seed values printed (full disclosure, incl. the seed-2 STF regression)
Writes exp/stats_scirep.json and prints a markdown summary.
"""
import os, json, math
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")

# seed-paired run groups (index = seed slot)
GROUPS = {
    "source-only": ["none_full", "none_s2", "none_s3"],
    "SAM":         ["sam005_full", "sam005_s2", "sam005_s3"],
    "PolarMix":    ["polarmix_s1", "polarmix_s2", "polarmix_s3"],
    "SAM+PolarMix": ["sampolar_s1", "sampolar_s2", "sampolar_s3"],
    "PointDR-insp": ["pointdr_s1", "pointdr_s2", "pointdr_s3"],
    "UniMix-insp":  ["wmix_s1", "wmix_s2", "wmix_s3"],
    "SWA":          ["swa_s1", "swa_s2", "swa_s3"],
    "cr1-source":   ["cr1_none_s1", "cr1_none_s2", "cr1_none_s3"],
    "cr1-SAM":      ["cr1_sam_s1", "cr1_sam_s2"],
}
PAIRED = [("SAM", "source-only"), ("SAM+PolarMix", "PolarMix"),
          ("SAM+PolarMix", "SAM"), ("PolarMix", "source-only"),
          ("cr1-SAM", "cr1-source")]

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def vals(runs, target):
    out = []
    for r in runs:
        p = os.path.join(EXP, r, "eval_all.json")
        if os.path.exists(p):
            out.append(json.load(open(p))[target]["mIoU"] * 100)
    return np.array(out)


def paired_stats(a, b):
    """a, b: same-length seed-paired arrays. Returns mean diff, sample SD, t, p, CI95."""
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    md = float(d.mean())
    if n < 2:
        return {"n": n, "mean_diff": md}
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else float("inf")
    # two-sided p via survival function of t-dist (scipy if present, else lookup-free approx)
    try:
        from scipy import stats as st
        p = float(2 * st.t.sf(abs(t), n - 1))
    except Exception:
        p = None
    ci = T975.get(n - 1, 1.96) * se
    return {"n": n, "mean_diff": round(md, 2), "sd_diff": round(sd, 2),
            "t": round(t, 2), "p_two_sided": (round(p, 3) if p is not None else None),
            "ci95": [round(md - ci, 2), round(md + ci, 2)],
            "per_seed_diffs": [round(float(x), 2) for x in d]}


def main():
    res = {"groups": {}, "paired": {}}
    print("## Group means (sample SD, ddof=1)\n")
    print("| method | n | KITTI mIoU | STF mIoU | per-seed K | per-seed S |")
    print("|---|---|---|---|---|---|")
    for g, runs in GROUPS.items():
        k, s = vals(runs, "kitti"), vals(runs, "stf")
        if len(k) == 0:
            continue
        row = {"runs": runs, "n": len(k),
               "kitti_mean": round(float(k.mean()), 2),
               "kitti_sd": (round(float(k.std(ddof=1)), 2) if len(k) > 1 else None),
               "stf_mean": round(float(s.mean()), 2),
               "stf_sd": (round(float(s.std(ddof=1)), 2) if len(s) > 1 else None),
               "kitti_per_seed": [round(float(x), 2) for x in k],
               "stf_per_seed": [round(float(x), 2) for x in s]}
        res["groups"][g] = row
        sd_k = f"±{row['kitti_sd']}" if row["kitti_sd"] is not None else ""
        sd_s = f"±{row['stf_sd']}" if row["stf_sd"] is not None else ""
        print(f"| {g} | {row['n']} | {row['kitti_mean']}{sd_k} | {row['stf_mean']}{sd_s} "
              f"| {row['kitti_per_seed']} | {row['stf_per_seed']} |")

    print("\n## Seed-paired comparisons (two-sided paired t, 95% CI)\n")
    print("| comparison | target | n | mean diff | 95% CI | p | per-seed diffs |")
    print("|---|---|---|---|---|---|---|")
    for a, b in PAIRED:
        if a not in res["groups"] or b not in res["groups"]:
            continue
        for tgt in ["kitti", "stf"]:
            va, vb = vals(GROUPS[a], tgt), vals(GROUPS[b], tgt)
            ps = paired_stats(va, vb)
            res["paired"][f"{a} - {b} ({tgt})"] = ps
            if "ci95" in ps:
                print(f"| {a} − {b} | {tgt} | {ps['n']} | {ps['mean_diff']} | "
                      f"[{ps['ci95'][0]}, {ps['ci95'][1]}] | {ps['p_two_sided']} | {ps['per_seed_diffs']} |")

    # BN recalibration deltas, if the test has produced them
    bn = {}
    for g in ["source-only", "SAM", "PolarMix", "SAM+PolarMix"]:
        rows = []
        for r in GROUPS[g]:
            p = os.path.join(EXP, r, "bn_recalib.json")
            if os.path.exists(p):
                j = json.load(open(p))
                # keep FULL precision here; round only at print time so group means
                # reproduce the paper's Table 3 exactly (mean-then-round)
                rows.append({"run": r,
                             "kitti_before": j["before"]["kitti"] * 100,
                             "kitti_after": j["after"]["kitti"] * 100,
                             "stf_before": j["before"]["stf"] * 100,
                             "stf_after": j["after"]["stf"] * 100})
        if rows:
            bn[g] = rows
    if bn:
        res["bn_recalib"] = bn
        print("\n## BN recalibration (common fixed source stream)\n")
        print("| method | n | KITTI before→after (mean) | STF before→after (mean) |")
        print("|---|---|---|---|")
        for g, rows in bn.items():
            kb = np.mean([r["kitti_before"] for r in rows]); ka = np.mean([r["kitti_after"] for r in rows])
            sb = np.mean([r["stf_before"] for r in rows]); sa = np.mean([r["stf_after"] for r in rows])
            print(f"| {g} | {len(rows)} | {kb:.2f} → {ka:.2f} ({ka-kb:+.2f}) | {sb:.2f} → {sa:.2f} ({sa-sb:+.2f}) |")
        for (a, b) in [("SAM", "source-only"), ("SAM+PolarMix", "PolarMix")]:
            if a in bn and b in bn:
                n = min(len(bn[a]), len(bn[b]))
                da_k = np.array([bn[a][i]["kitti_after"] for i in range(n)]) - np.array([bn[b][i]["kitti_after"] for i in range(n)])
                da_s = np.array([bn[a][i]["stf_after"] for i in range(n)]) - np.array([bn[b][i]["stf_after"] for i in range(n)])
                db_k = np.array([bn[a][i]["kitti_before"] for i in range(n)]) - np.array([bn[b][i]["kitti_before"] for i in range(n)])
                db_s = np.array([bn[a][i]["stf_before"] for i in range(n)]) - np.array([bn[b][i]["stf_before"] for i in range(n)])
                print(f"\n{a} − {b} gap: KITTI before {db_k.mean():+.2f} → after {da_k.mean():+.2f}; "
                      f"STF before {db_s.mean():+.2f} → after {da_s.mean():+.2f}")

    json.dump(res, open(os.path.join(EXP, "stats_scirep.json"), "w"), indent=2)
    print(f"\nwritten: {os.path.join(EXP, 'stats_scirep.json')}")


if __name__ == "__main__":
    main()
