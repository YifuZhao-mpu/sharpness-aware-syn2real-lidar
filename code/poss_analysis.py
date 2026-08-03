"""Frozen analysis of the sealed SemanticPOSS evaluation (per POSS_FREEZE_MANIFEST.md):
per-method mean +- sample SD over seeds; seed-paired diffs (SAM - source-only,
SAM+PolarMix - PolarMix) with paired t and 95% CI; full per-checkpoint listing.
Written before the sealed run completed; runs once on poss_summary.json."""
import os, json, math
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
S = json.load(open(os.path.join(EXP, "poss_summary.json")))

GROUPS = {
    "source-only": ["none_full", "none_s2", "none_s3"],
    "SAM": ["sam005_full", "sam005_s2", "sam005_s3"],
    "PolarMix-style": ["polarmix_s1", "polarmix_s2", "polarmix_s3"],
    "SAM+PolarMix": ["sampolar_s1", "sampolar_s2", "sampolar_s3"],
    "SWA": ["swa_s1", "swa_s2", "swa_s3"],
    "PointDR-inspired": ["pointdr_s1", "pointdr_s2", "pointdr_s3"],
    "UniMix-inspired": ["wmix_s1", "wmix_s2", "wmix_s3"],
    "cr1-source": ["cr1_none_s1", "cr1_none_s2", "cr1_none_s3"],
    "cr1-SAM": ["cr1_sam_s1", "cr1_sam_s2"],
    "Sensing-SAM": ["samsens_s1", "samsens_s2"],
}
PAIRED = [("SAM", "source-only"), ("SAM+PolarMix", "PolarMix-style"),
          ("PolarMix-style", "source-only"), ("SAM+PolarMix", "SAM"),
          ("cr1-SAM", "cr1-source")]
T975 = {1: 12.706, 2: 4.303}


def arr(runs, key):
    return np.array([S[r][key] for r in runs if r in S and S[r].get(key) is not None])


def paired(a, b):
    n = min(len(a), len(b)); d = a[:n] - b[:n]
    md = float(d.mean())
    if n < 2:
        return f"n={n} diff={md:+.2f}"
    sd = float(d.std(ddof=1)); se = sd / math.sqrt(n)
    t = md / se if se > 0 else float("inf")
    try:
        from scipy import stats as st
        p = float(2 * st.t.sf(abs(t), n - 1))
    except Exception:
        p = None
    ci = T975.get(n - 1, 1.96) * se
    return (f"n={n} diff={md:+.2f} CI95=[{md-ci:+.2f},{md+ci:+.2f}] "
            f"p={p if p is None else round(p,3)} per-seed={[round(float(x),2) for x in d]}")


print("## Sealed SemanticPOSS (11-class common space) — per-method summary\n")
print("| method | n | as-is mIoU | BN-recalibrated mIoU | per-seed (as-is) |")
print("|---|---|---|---|---|")
for g, runs in GROUPS.items():
    a = arr(runs, "asis"); b = arr(runs, "bn_recalib")
    if len(a) == 0:
        continue
    sa = f"±{a.std(ddof=1):.2f}" if len(a) > 1 else ""
    sb = f"±{b.std(ddof=1):.2f}" if len(b) > 1 else ""
    print(f"| {g} | {len(a)} | {a.mean():.2f}{sa} | {b.mean():.2f}{sb} "
          f"| {[round(float(x),2) for x in a]} |")

print("\n## Frozen paired contrasts (as-is checkpoints)\n")
for x, y in PAIRED:
    print(f"{x} − {y}: {paired(arr(GROUPS[x],'asis'), arr(GROUPS[y],'asis'))}")
print("\n## Same contrasts under common BN recalibration\n")
for x, y in PAIRED:
    print(f"{x} − {y}: {paired(arr(GROUPS[x],'bn_recalib'), arr(GROUPS[y],'bn_recalib'))}")

print("\n## All checkpoints (as-is | BN-recal)\n")
for r in sorted(S, key=lambda r: -S[r]["asis"]):
    br = S[r].get("bn_recalib")
    print(f"  {r:18s} {S[r]['asis']:5.2f} | {br if br is None else round(br,2)}")
