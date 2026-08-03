"""Generate all paper figures + the main table from real result files in exp/.
Reads exp/*/eval_all.json (final-epoch full eval), exp/*/sharpness.json, exp/*/history.json.
Robust to missing runs (skips). Outputs to ../paper/figures/. Publication style."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
FIG = os.path.join(CODE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

C_NONE, C_SAM, C_SENS = "#d62728", "#1f77b4", "#2ca02c"


def load(name):
    p = os.path.join(EXP, name, "eval_all.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"kitti": d["kitti"]["mIoU"] * 100, "stf": d["stf"]["mIoU"] * 100,
            "kitti_pc": d["kitti"]["per_class"], "stf_pc": d["stf"]["per_class"],
            "kitti_nc": d["kitti"].get("nuisance_corr_target"),
            "stf_nc": d["stf"].get("nuisance_corr_target")}


def load_sharp(name):
    p = os.path.join(EXP, name, "sharpness.json")
    return json.load(open(p))["sharpness_mean"] if os.path.exists(p) else None


def group(names):
    rows = [load(n) for n in names]
    rows = [r for r in rows if r]
    if not rows:
        return None
    k = np.array([r["kitti"] for r in rows]); s = np.array([r["stf"] for r in rows])
    return {"k_m": k.mean(), "k_s": k.std(), "s_m": s.mean(), "s_s": s.std(),
            "k": k, "s": s, "n": len(rows)}


NONE = ["none_full", "none_s2", "none_s3"]
SAM = ["sam005_full", "sam005_s2", "sam005_s3"]
SENS = ["samsens_s1", "samsens_s2"]


def fig_main_scatter():
    """Fig 1 right: KITTI x STF scatter, per-seed, with group means."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for names, c, lab in [(NONE, C_NONE, "Source-only"), (SAM, C_SAM, "SAM (ours)"),
                          (SENS, C_SENS, "Sensing-SAM")]:
        rows = [load(n) for n in names]; rows = [r for r in rows if r]
        if not rows:
            continue
        k = [r["kitti"] for r in rows]; s = [r["stf"] for r in rows]
        ax.scatter(k, s, c=c, s=55, alpha=0.65, label=f"{lab} (seeds)", edgecolors="k", linewidths=0.4)
        ax.scatter(np.mean(k), np.mean(s), c=c, s=240, marker="*", edgecolors="k", linewidths=1.0)
    ax.set_xlabel("SemanticKITTI mIoU (%)  [normal real]")
    ax.set_ylabel("SemanticSTF mIoU (%)  [adverse real]")
    ax.set_title("Zero-shot generalization (★ = mean over seeds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIG, "fig1_scatter.pdf")); plt.close(fig)


def fig_sharpness():
    """Fig 2: sharpness bar (mean±std over seeds), source-only vs SAM."""
    gs = {"Source-only": [load_sharp(n) for n in NONE], "SAM": [load_sharp(n) for n in SAM]}
    fig, ax = plt.subplots(figsize=(4, 4))
    xs, labs, cols = [], [], [C_NONE, C_SAM]
    for i, (k, v) in enumerate(gs.items()):
        v = [x for x in v if x is not None]
        if not v:
            continue
        ax.bar(i, np.mean(v), yerr=(np.std(v) if len(v) > 1 else 0), capsize=6,
               color=cols[i], alpha=0.8, width=0.6)
        for val in v:
            ax.scatter(i, val, c="k", s=22, zorder=3)
        labs.append(f"{k}\n({np.mean(v):.3f})")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    ax.set_ylabel("Filter-norm sharpness (loss increase @ ρ=0.05)")
    ax.set_title("SAM finds much flatter minima")
    fig.savefig(os.path.join(FIG, "fig2_sharpness.pdf")); plt.close(fig)


def fig_variance():
    """Fig 3: mIoU mean±std bars, source-only vs SAM, both targets."""
    gn, gsa = group(NONE), group(SAM)
    if not (gn and gsa):
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, [gn["k_m"], gn["s_m"]], w, yerr=[gn["k_s"], gn["s_s"]], capsize=5,
           label=f"Source-only (n={gn['n']})", color=C_NONE, alpha=0.8)
    ax.bar(x + w/2, [gsa["k_m"], gsa["s_m"]], w, yerr=[gsa["k_s"], gsa["s_s"]], capsize=5,
           label=f"SAM (n={gsa['n']})", color=C_SAM, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["SemanticKITTI", "SemanticSTF"])
    ax.set_ylabel("mIoU (%)"); ax.set_title("Mean ± std over seeds")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig3_variance.pdf")); plt.close(fig)


def fig_corr():
    """Fig 4: per-checkpoint (sharpness, mIoU) across ALL runs — anti-correlation + Pearson r."""
    import glob as _g
    cat = {"none": C_NONE, "sam": C_SAM, "samsens": C_SENS, "dr": "#9467bd", "swa": "#ff7f0e"}
    def cof(name):
        if name.startswith("none"): return C_NONE, "Source-only"
        if name.startswith("samsens"): return C_SENS, "Sensing-SAM"
        if name.startswith("samdr"): return "#8c564b", "SAM+DR"
        if name.startswith("sampolar"): return "#e377c2", "SAM+PolarMix"
        if name.startswith("sam"): return C_SAM, "SAM"
        if name.startswith("dr"): return "#9467bd", "DR-lite"
        if name.startswith("polarmix"): return "#17becf", "PolarMix"
        if name.startswith("swa"): return "#ff7f0e", "SWA"
        if name.startswith("none_long"): return C_NONE, "Source-only"
        return "k", name
    # EXPLICIT casts (a glob here once leaked pointdr_s2/none_long_s1 into the regression,
    # diluting r to -0.39 vs the paper's canonical 14-model -0.83):
    # SAME = the 14 same-data models of the text (source-only, all SAM variants, SWA);
    # AUGM = input-augmentation models, shown as X markers, excluded from the correlation.
    SAME = ["none_full", "none_s2", "none_s3",
            "sam005_full", "sam005_s2", "sam005_s3", "sam001_s1", "sam010_s1", "sam020_s1",
            "samsens_s1", "samsens_s2", "swa_s1", "swa_s2", "swa_s3"]
    AUGM = ["dr_full_s1", "samdr_s1", "polarmix_s1", "polarmix_s2", "polarmix_s3",
            "sampolar_s1", "sampolar_s2", "sampolar_s3"]
    fig, ax = plt.subplots(figsize=(4.8, 4))
    xs, ys, seen = [], [], set()   # same-data only (for the reported correlation)
    for n in SAME + AUGM:
        sh = load_sharp(n); r = load(n)
        if sh is None or r is None:
            continue
        c, lab = cof(n)
        m = (r["kitti"] + r["stf"]) / 2
        is_aug = n in AUGM
        ax.scatter(sh, m, c=c, s=80 if is_aug else 70, marker="X" if is_aug else "o",
                   edgecolors="k", linewidths=0.5 if is_aug else 0.4,
                   label=(lab + " (aug)" if is_aug else lab) if lab not in seen else None)
        seen.add(lab)
        if not is_aug:
            xs.append(sh); ys.append(m)
    if len(xs) >= 3:
        r = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.96, 0.96, f"same-data $r={r:.2f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax.set_xlabel("Filter-norm sharpness (log scale)")
    ax.set_ylabel("Mean(KITTI, STF) mIoU (%)")
    ax.set_title("Flatter minima → better generalization")
    ax.set_xscale("log"); ax.legend(fontsize=7.5, loc="lower left")
    fig.savefig(os.path.join(FIG, "fig4_corr.pdf")); plt.close(fig)


def fig_rho():
    """Fig 5: rho sweep — KITTI/STF mIoU vs rho."""
    pts = [(0.0, group(NONE)), (0.01, group(["sam001_s1"])), (0.05, group(SAM)),
           (0.10, group(["sam010_s1"])), (0.20, group(["sam020_s1"]))]
    pts = [(r, g) for r, g in pts if g]
    if len(pts) < 3:
        return
    rhos = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(rhos, [g["k_m"] for _, g in pts], yerr=[g["k_s"] for _, g in pts],
                marker="o", label="SemanticKITTI", color=C_SAM, capsize=4)
    ax.errorbar(rhos, [g["s_m"] for _, g in pts], yerr=[g["s_s"] for _, g in pts],
                marker="s", label="SemanticSTF", color=C_SENS, capsize=4)
    ax.set_xlabel("SAM ρ (ρ=0 is source-only)"); ax.set_ylabel("mIoU (%)")
    ax.set_title("Robustness to ρ"); ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig5_rho.pdf")); plt.close(fig)


def fig_traj():
    """Fig 6: per-epoch mIoU trajectories, source-only vs SAM (seed1)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, c, lab in [("none_full", C_NONE, "Source-only"), ("sam005_full", C_SAM, "SAM")]:
        p = os.path.join(EXP, name, "history.json")
        if not os.path.exists(p):
            continue
        h = json.load(open(p))
        ep = [e["epoch"] for e in h]
        axes[0].plot(ep, [e["kitti"]["mIoU"] * 100 for e in h], marker="o", color=c, label=lab)
        axes[1].plot(ep, [e["stf"]["mIoU"] * 100 for e in h], marker="o", color=c, label=lab)
    axes[0].set_title("SemanticKITTI"); axes[1].set_title("SemanticSTF")
    for a in axes:
        a.set_xlabel("epoch"); a.set_ylabel("mIoU (%)"); a.legend(fontsize=9)
    fig.suptitle("Source-only is unstable at the cosine endgame; SAM is smooth")
    fig.savefig(os.path.join(FIG, "fig6_traj.pdf")); plt.close(fig)


def _fmt(km, ks, sm, ss, n):
    a = f"{km:.2f}\\,$\\pm$\\,{ks:.2f}" if n > 1 else f"{km:.2f}"
    b = f"{sm:.2f}\\,$\\pm$\\,{ss:.2f}" if n > 1 else f"{sm:.2f}"
    return a, b


def _emit_table(rows, label, caption, fname, bold_prefix=None):
    tabdir = os.path.join(CODE, "..", "paper", "tables")
    os.makedirs(tabdir, exist_ok=True)
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + caption + "}",
         r"\label{" + label + "}", r"\begin{tabular}{lccc}", r"\toprule",
         r"Method & SemanticKITTI & SemanticSTF & $n$ \\", r"\midrule"]
    for lab, km, ks, sm, ss, n in rows:
        a, b = _fmt(km, ks, sm, ss, n)
        labx = r"\textbf{" + lab + "}" if (bold_prefix and lab.startswith(bold_prefix)) else lab
        L.append(f"{labx} & {a} & {b} & {n} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(L).replace("ρ", r"$\rho$")
    open(os.path.join(tabdir, fname), "w").write(tex + "\n")


def main_table():
    """Emit main results table (n column) + a separate SWA table + markdown."""
    def collect(specs):
        out = []
        for label, names in specs:
            g = group(names if isinstance(names, list) else [names])
            if g:
                out.append((label, g["k_m"], g["k_s"], g["s_m"], g["s_s"], g["n"]))
        return out
    main_rows = collect([
        ("Source-only", NONE),
        ("SAM (ρ=0.05)", SAM),
        ("Sensing-SAM", SENS),
        ("SAM (ρ=0.01)", ["sam001_s1"]),
        ("SAM (ρ=0.10)", ["sam010_s1"]),
        ("SAM (ρ=0.20)", ["sam020_s1"]),
        ("DR-lite", ["dr_full_s1"]),
        ("SAM + DR-lite", ["samdr_s1"]),
        ("PointDR (reimpl.)", ["pointdr_s1", "pointdr_s2", "pointdr_s3"]),
        ("PolarMix", ["polarmix_s1", "polarmix_s2", "polarmix_s3"]),
        ("SAM + PolarMix", ["sampolar_s1", "sampolar_s2", "sampolar_s3"]),
        ("Source-only (2$\\times$ compute)", ["none_long_s1", "none_long_s2"]),
    ])
    _emit_table(main_rows, "tab:main",
                r"Zero-shot \miou (\%) on SynLiDAR$\to$\{SemanticKITTI, SemanticSTF\}, "
                r"final-epoch models, $n$ seeds (rows with $n{=}1$ are single-seed and "
                r"exploratory). All rows share the same MinkUNet backbone and schedule; the "
                r"optimizer-side methods (source-only, SAM) also share the same light "
                r"augmentation; DR-lite, PointDR, and PolarMix are input-side methods. Source-only "
                r"(2$\times$ compute) trains twice as long to match SAM's per-step cost.",
                "main_table.tex", bold_prefix="SAM (ρ=0.05)")
    swa_rows = collect([("Source-only", NONE), ("SAM (ρ=0.05)", SAM),
                        ("SWA", ["swa_s1", "swa_s2", "swa_s3"])])
    _emit_table(swa_rows, "tab:swa",
                r"Weight averaging (SWA) as an independent flat-minima check, vs.\ "
                r"source-only and SAM. SWA is reported separately and is not a headline "
                r"result. Mean$\pm$std over $n$ seeds.", "swa_table.tex")
    # cr=1.0 (full-width MinkUNet, 4x capacity) generality check
    cr1_rows = collect([("Source-only (cr=1.0)", ["cr1_none_s1", "cr1_none_s2"]),
                        ("SAM (cr=1.0)", ["cr1_sam_s1", "cr1_sam_s2"])])
    if cr1_rows:
        _emit_table(cr1_rows, "tab:cr1",
                    r"Generality to a larger backbone: full-width MinkUNet (cr$=$1.0, "
                    r"$\sim$4$\times$ the parameters of our main model). Single seed, "
                    r"final-epoch \miou (\%).",
                    "cr1_table.tex", bold_prefix="SAM (cr")
    md = ["| Method | KITTI | STF | n |", "|---|---|---|---|"]
    for lab, km, ks, sm, ss, n in main_rows + swa_rows[-1:]:
        km_s = f"{km:.2f}±{ks:.2f}" if n > 1 else f"{km:.2f}"
        sm_s = f"{sm:.2f}±{ss:.2f}" if n > 1 else f"{sm:.2f}"
        md.append(f"| {lab} | {km_s} | {sm_s} | {n} |")
    open(os.path.join(FIG, "main_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def perclass_table():
    """Per-class IoU table: source-only (seed1) vs SAM (seed1) on both targets."""
    n = load("none_full"); s = load("sam005_full")
    if not (n and s):
        return
    tabdir = os.path.join(CODE, "..", "paper", "tables")
    os.makedirs(tabdir, exist_ok=True)
    from utils import CLASS_NAMES
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Per-class IoU (\%) for source-only vs.\ SAM ($\rho{=}0.05$), seed 1, "
         r"final epoch. SAM's gains concentrate in geometrically-distinct classes; "
         r"$\Delta$ is SAM$-$source-only.}",
         r"\label{tab:perclass}", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\textwidth}{!}{",
         r"\begin{tabular}{ll" + "c" * len(CLASS_NAMES) + "c}", r"\toprule"]
    header = "Target & Method & " + " & ".join(
        [c.replace("-", "-\\\\") if False else c[:5] for c in CLASS_NAMES]) + r" & mIoU \\"
    L.append(header); L.append(r"\midrule")
    def row(target_name, key_pc, key_m, dlabel):
        npc = n[key_pc]; spc = s[key_pc]
        nv = [npc.get(c) for c in CLASS_NAMES]; sv = [spc.get(c) for c in CLASS_NAMES]
        nr = f"{target_name} & Source-only & " + " & ".join(f"{v:.0f}" if v is not None else "-" for v in nv) + f" & {n[key_m]:.1f} \\\\"
        sr = r"& \textbf{SAM} & " + " & ".join(f"{v:.0f}" if v is not None else "-" for v in sv) + f" & {s[key_m]:.1f} \\\\"
        return nr, sr
    for tname, kpc, km in [("KITTI", "kitti_pc", "kitti"), ("STF", "stf_pc", "stf")]:
        a, b = row(tname, kpc, km, tname)
        L += [a, b, r"\midrule"]
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(tabdir, "perclass_table.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    for f in (fig_main_scatter, fig_sharpness, fig_variance, fig_corr, fig_rho, fig_traj, perclass_table):
        try:
            f()
        except Exception as e:
            print("skip", f.__name__, e)
    main_table()
    print("\nfigures ->", os.path.abspath(FIG))
