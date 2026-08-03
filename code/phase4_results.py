"""Aggregate phase-4 results vs baselines and render the SOTA verdict.
Run anytime: python phase4_results.py   (uses whatever eval_all.json exist so far).
Reports per-recipe mean+-std (full-eval), SAM's additive value, and whether STF clearly
exceeds (a) published UniMix SOTA 23.4 and (b) a protocol-adjusted threshold ~26."""
import os, json, glob
import numpy as np
from collections import defaultdict

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")

# recipe -> human label (what augmentation/optimizer routes are on)
LABEL = {
    "none": "Source-only", "sam005": "SAM", "polarmix": "PolarMix",
    "sampolar": "SAM+PolarMix", "pointdr": "PointDR(reimpl)",
    "polarlm": "PolarMix+LaserMix", "sampolarlm": "SAM+PolarMix+LaserMix",
    "wmix": "PolarMix+LaserMix+Weather (UniMix-style)",
    "samwmix": "SAM+PolarMix+LaserMix+Weather",
    "sampolarw": "SAM+PolarMix+Weather", "lasermix": "LaserMix", "weather": "Weather",
}
PUBLISHED_SOTA_STF = 23.4   # UniMix CVPR'24 (verified)
PROTO_ADJ_STF = 26.0        # rough protocol-adjusted UniMix (our PointDR reimpl runs ~+2.3 over published)


def load():
    g = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(EXP, "*/eval_all.json"))):
        name = os.path.basename(os.path.dirname(f))
        if name.startswith(("pilot_", "_smoke")):
            continue
        base = name.rsplit("_s", 1)[0] if ("_s" in name and name.rsplit("_s", 1)[1].isdigit()) else name
        try:
            d = json.load(open(f))
            sh = None
            shf = os.path.join(os.path.dirname(f), "sharpness.json")
            if os.path.exists(shf):
                sh = json.load(open(shf)).get("sharpness_mean")
            g[base].append((name, d["kitti"]["mIoU"] * 100, d["stf"]["mIoU"] * 100, sh))
        except Exception:
            pass
    return g


def agg(runs):
    ks = np.array([r[1] for r in runs]); ss = np.array([r[2] for r in runs])
    return ks.mean(), ks.std(), ss.mean(), ss.std(), len(runs)


def main():
    g = load()
    order = ["none", "sam005", "pointdr", "polarmix", "sampolar", "lasermix", "weather",
             "polarlm", "sampolarlm", "sampolarw", "wmix", "samwmix"]
    print(f"{'recipe':<42}{'KITTI':>14}{'STF':>14}{'n':>3}  sharp")
    print("-" * 92)
    res = {}
    for base in order:
        if base not in g:
            continue
        km, ks, sm, ss, n = agg(g[base])
        res[base] = (km, sm, n)
        shs = [r[3] for r in g[base] if r[3] is not None]
        shstr = f"{np.mean(shs):.3f}" if shs else "-"
        kstr = f"{km:.2f}+-{ks:.2f}" if n > 1 else f"{km:.2f}"
        sstr = f"{sm:.2f}+-{ss:.2f}" if n > 1 else f"{sm:.2f}"
        star = "  <-- " + LABEL.get(base, base) if base in ("samwmix", "sampolarlm", "sampolarw") else ""
        print(f"{LABEL.get(base,base):<42}{kstr:>14}{sstr:>14}{n:>3}  {shstr}{star}")
    print("-" * 92)
    # SAM additive value on the full pipeline
    if "samwmix" in res and "wmix" in res:
        dK = res["samwmix"][0] - res["wmix"][0]; dS = res["samwmix"][1] - res["wmix"][1]
        print(f"SAM's additive value on full pipeline (samwmix - wmix): {dK:+.2f} KITTI / {dS:+.2f} STF")
    if "sampolarlm" in res and "polarlm" in res:
        dK = res["sampolarlm"][0] - res["polarlm"][0]; dS = res["sampolarlm"][1] - res["polarlm"][1]
        print(f"SAM's additive value on mixing-only (sampolarlm - polarlm): {dK:+.2f} K / {dS:+.2f} S")
    # best STF achieved
    best = max(((b, res[b][1], res[b][2]) for b in res), key=lambda x: x[1], default=None)
    if best:
        b, sstf, n = best
        print(f"\nBest STF: {LABEL.get(b,b)} = {sstf:.2f} (n={n})")
        print(f"  vs published UniMix SOTA {PUBLISHED_SOTA_STF}:  {'EXCEEDS by +%.1f' % (sstf-PUBLISHED_SOTA_STF) if sstf>PUBLISHED_SOTA_STF else 'below'}")
        print(f"  vs protocol-adjusted ~{PROTO_ADJ_STF}:        {'CLEARS by +%.1f' % (sstf-PROTO_ADJ_STF) if sstf>PROTO_ADJ_STF else 'NOT yet (%.1f short)' % (PROTO_ADJ_STF-sstf)}")


if __name__ == "__main__":
    main()
