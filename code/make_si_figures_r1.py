"""Supplementary Figs S1 (SAM radius sweep) and S2 (training trajectories), Revision 1.

Same data and layout as fig_rho / fig_traj in make_figures.py, with three changes made for
the reviewers' consistency requests: (i) the in-figure titles, which stated conclusions
("Robustness to rho"; "Source-only is unstable ...; SAM is smooth"), are removed and the
captions carry the description; (ii) error bars use the sample SD (ddof=1), as everywhere
else in the article (make_figures.py used the population SD); (iii) labels follow the
manuscript's terminology. Writes paper/srep/figures/fig5_rho.pdf and fig6_traj.pdf."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
OUT = os.path.join(CODE, "..", "paper", "srep", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_NONE, C_SAM, C_SENS = "#d62728", "#1f77b4", "#2ca02c"


def miou(run, tgt):
    return json.load(open(os.path.join(EXP, run, "eval_all.json")))[tgt]["mIoU"] * 100


def mean_sd(runs, tgt):
    v = np.array([miou(r, tgt) for r in runs])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fig_rho():
    pts = [(0.0, ["none_full", "none_s2", "none_s3"]), (0.01, ["sam001_s1"]),
           (0.05, ["sam005_full", "sam005_s2", "sam005_s3"]), (0.10, ["sam010_s1"]),
           (0.20, ["sam020_s1"])]
    rhos = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(5, 4))
    for tgt, lab, col, mk in (("kitti", "SemanticKITTI", C_SAM, "o"), ("stf", "SemanticSTF", C_SENS, "s")):
        ms = [mean_sd(r, tgt) for _, r in pts]
        ax.errorbar(rhos, [m for m, _ in ms], yerr=[s for _, s in ms], marker=mk, label=lab,
                    color=col, capsize=4)
    ax.set_xlabel("SAM radius ρ (ρ = 0 is source-only)")
    ax.set_ylabel("mIoU (%)")
    ax.legend(fontsize=10)
    fig.savefig(os.path.join(OUT, "fig5_rho.pdf")); plt.close(fig)


def fig_traj():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, c, lab in (("none_full", C_NONE, "Source-only"), ("sam005_full", C_SAM, "SAM")):
        h = json.load(open(os.path.join(EXP, name, "history.json")))
        ep = [e["epoch"] for e in h]
        axes[0].plot(ep, [e["kitti"]["mIoU"] * 100 for e in h], marker="o", color=c, label=lab)
        axes[1].plot(ep, [e["stf"]["mIoU"] * 100 for e in h], marker="o", color=c, label=lab)
    axes[0].set_title("SemanticKITTI"); axes[1].set_title("SemanticSTF")
    for a in axes:
        a.set_xlabel("Epoch"); a.set_ylabel("mIoU (%)"); a.legend(fontsize=10)
    fig.savefig(os.path.join(OUT, "fig6_traj.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_rho(); fig_traj()
    print("wrote", os.path.join(OUT, "fig5_rho.pdf"), "and fig6_traj.pdf")
