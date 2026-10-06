"""Figure 2: perturbation-probe sharpness (sharpness2) vs mean development-target mIoU.

Filled circles = frozen 13-checkpoint clean-input cohort (the only points entering the
correlation); hollow markers = methods trained on modified inputs (context only).
Colour encodes the optimizer family (blue: SAM-containing; orange: no SAM; teal: SWA);
marker shape identifies the context method, and PolarMix-style / SAM + PolarMix-style
share a shape so the effect of adding SAM to that augmentation reads directly.
The three hues pass an all-pairs colour-vision-deficiency check (OKLab dE >= 9.2).

Revision 1 (2026-10): larger axis, tick and legend text for print legibility (reviewer
request), legend moved below the axes so it no longer covers data, method names
harmonized with the manuscript. Data, cohort and statistics are unchanged.
Writes paper/srep/figures/fig_sharp2.pdf"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
OUT = os.path.join(os.path.dirname(CODE), "paper", "srep", "figures")
os.makedirs(OUT, exist_ok=True)

SAM_BLUE, NOSAM_ORANGE, SWA_TEAL = "#2a78d6", "#eb6834", "#1baf7a"

# clean-input cohort (filled circles); entries: label -> (runs, colour)
COHORT = {"Source-only": (["none_full", "none_s2", "none_s3", "none_long_s1"], NOSAM_ORANGE),
          "SAM (\u03c1 = 0.01\u20130.2)": (["sam001_s1", "sam005_full", "sam005_s2", "sam005_s3",
                                          "sam010_s1", "sam020_s1"], SAM_BLUE),
          "SWA": (["swa_s1", "swa_s2", "swa_s3"], SWA_TEAL)}
# modified-input context methods (hollow markers); label -> (runs, colour, marker)
EXTRA = {"PolarMix-style": (["polarmix_s1", "polarmix_s2", "polarmix_s3"], NOSAM_ORANGE, "s"),
         "SAM + PolarMix-style": (["sampolar_s1", "sampolar_s2", "sampolar_s3"], SAM_BLUE, "s"),
         "PointDR-inspired": (["pointdr_s1", "pointdr_s2", "pointdr_s3"], NOSAM_ORANGE, "D"),
         "DR-lite": (["dr_full_s1"], NOSAM_ORANGE, "v"),
         "Sensing-SAM": (["samsens_s1", "samsens_s2"], SAM_BLUE, "^")}


def load(run):
    s = json.load(open(os.path.join(EXP, run, "sharpness2.json")))["sharpness_mean"]
    e = json.load(open(os.path.join(EXP, run, "eval_all.json")))
    return s, (e["kitti"]["mIoU"] + e["stf"]["mIoU"]) / 2 * 100


plt.rcParams.update({"font.size": 11, "axes.labelsize": 12.5, "xtick.labelsize": 11.5,
                     "ytick.labelsize": 11.5, "legend.fontsize": 10.5,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.4, 5.4))
xs, ys, handles_c, handles_x = [], [], [], []
for lab, (runs, col) in COHORT.items():
    v = [load(r) for r in runs]
    ax.scatter([x[0] for x in v], [x[1] for x in v], s=95, marker="o", c=col,
               edgecolors="white", linewidths=1.2, zorder=3)
    xs += [x[0] for x in v]; ys += [x[1] for x in v]
    handles_c.append(Line2D([], [], ls="none", marker="o", ms=9.5, mfc=col, mec="white",
                            mew=1.2, label=lab))
for lab, (runs, col, mk) in EXTRA.items():
    v = [load(r) for r in runs]
    ax.scatter([x[0] for x in v], [x[1] for x in v], s=78, marker=mk, facecolors="white",
               edgecolors=col, linewidths=1.9, zorder=2)
    handles_x.append(Line2D([], [], ls="none", marker=mk, ms=8.5, mfc="white", mec=col,
                            mew=1.9, label=lab))

r = float(np.corrcoef(xs, ys)[0, 1])
rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
rs = float(np.corrcoef(rx, ry)[0, 1])
ax.set_xscale("log")
ax.set_xlabel(r"Perturbation-probe sharpness $\mathcal{S}$ (log scale)")
ax.set_ylabel("Mean development-target mIoU (%)")
minus = lambda v: f"{v:.2f}".replace("-", "\u2212")  # typographic minus sign
ax.text(0.98, 0.97, f"Clean-input cohort (filled circles, n = {len(xs)}):\n"
        f"Pearson r = {minus(r)}, Spearman " + r"$\rho_s$" + f" = {minus(rs)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="#0b0b0b",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c9c8c4", lw=0.8))
ax.grid(color="#e4e3df", lw=0.8, which="major")
ax.grid(color="#f0efec", lw=0.6, which="minor")
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
# legend below the axes: left block = cohort, right block = context methods
leg1 = fig.legend(handles=handles_c, title="Clean-input cohort", loc="upper left",
                  bbox_to_anchor=(0.10, 0.215), frameon=False, title_fontsize=10.5,
                  alignment="left", handletextpad=0.3, labelspacing=0.35)
leg2 = fig.legend(handles=handles_x, title="Modified-input methods (context only)",
                  loc="upper left", bbox_to_anchor=(0.45, 0.215), ncol=2, frameon=False,
                  title_fontsize=10.5, alignment="left", handletextpad=0.3,
                  columnspacing=1.0, labelspacing=0.35)
fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.33)
fig.savefig(os.path.join(OUT, "fig_sharp2.pdf"))
fig.savefig(os.path.join(OUT, "fig_sharp2_preview.png"), dpi=150)
print("saved", os.path.join(OUT, "fig_sharp2.pdf"), f"r={r:.3f} spearman={rs:.3f}")
