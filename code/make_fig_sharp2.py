"""Figure: rigorous sharpness probe (sharpness2) vs mean development-target mIoU.
Circles = frozen 13-checkpoint clean-input cohort; crosses = modified-input methods
(context only). Writes paper/srep/figures/fig_sharp2.pdf"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
OUT = os.path.join(os.path.dirname(CODE), "paper", "srep", "figures")
os.makedirs(OUT, exist_ok=True)

COHORT = {"Source-only": ["none_full", "none_s2", "none_s3", "none_long_s1"],
          "SAM": ["sam001_s1", "sam005_full", "sam005_s2", "sam005_s3", "sam010_s1", "sam020_s1"],
          "SWA": ["swa_s1", "swa_s2", "swa_s3"]}
EXTRA = {"PolarMix-style": ["polarmix_s1", "polarmix_s2", "polarmix_s3"],
         "SAM+PolarMix": ["sampolar_s1", "sampolar_s2", "sampolar_s3"],
         "PointDR-inspired": ["pointdr_s1", "pointdr_s2", "pointdr_s3"],
         "Sensing-SAM": ["samsens_s1", "samsens_s2"],
         "DR-lite": ["dr_full_s1"]}
COLORS = {"Source-only": "#d62728", "SAM": "#1f77b4", "SWA": "#ff7f0e",
          "PolarMix-style": "#17becf", "SAM+PolarMix": "#e377c2",
          "PointDR-inspired": "#7f7f7f", "Sensing-SAM": "#2ca02c", "DR-lite": "#9467bd"}


def load(run):
    s = json.load(open(os.path.join(EXP, run, "sharpness2.json")))["sharpness_mean"]
    e = json.load(open(os.path.join(EXP, run, "eval_all.json")))
    return s, (e["kitti"]["mIoU"] + e["stf"]["mIoU"]) / 2 * 100


fig, ax = plt.subplots(figsize=(5.6, 4.2))
xs, ys = [], []
for lab, runs in COHORT.items():
    v = [load(r) for r in runs]
    ax.scatter([x[0] for x in v], [x[1] for x in v], s=75, marker="o",
               c=COLORS[lab], edgecolors="k", linewidths=0.5, label=lab, zorder=3)
    xs += [x[0] for x in v]; ys += [x[1] for x in v]
for lab, runs in EXTRA.items():
    v = [load(r) for r in runs]
    ax.scatter([x[0] for x in v], [x[1] for x in v], s=70, marker="X",
               c=COLORS[lab], edgecolors="k", linewidths=0.4, label=lab, zorder=2, alpha=0.85)

r = float(np.corrcoef(xs, ys)[0, 1])
rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
rs = float(np.corrcoef(rx, ry)[0, 1])
ax.set_xscale("log")
ax.set_xlabel(r"perturbation-probe sharpness $\mathcal{S}$ (log scale)")
ax.set_ylabel("mean development-target mIoU (%)")
ax.set_title(f"clean-input cohort (circles, n=13): Pearson r={r:.2f}, Spearman={rs:.2f}",
             fontsize=9.5)
ax.legend(fontsize=7.5, ncol=2, frameon=True, loc="lower left")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sharp2.pdf"))
print("saved", os.path.join(OUT, "fig_sharp2.pdf"), f"r={r:.3f} spearman={rs:.3f}")
