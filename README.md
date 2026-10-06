# Sharpness-aware training for synthetic-to-real LiDAR semantic segmentation

Code, raw per-run evaluation files, and analysis scripts for the manuscript:

> **Controlled empirical characterization of sharpness aware minimization for synthetic
> to real transfer in LiDAR semantic segmentation**
> Yifu Zhao, Xiaofan Zou, Junhao Wei, Yanxiao Li, Haochen Li, Baili Lu,
> Sio-Kei Im, Yapeng Wang\*, Xu Yang (\*corresponding: yapengwang@mpu.edu.mo)
> Faculty of Applied Sciences, Macao Polytechnic University

This is a reproducibility archive, not a framework. Every number, table, and figure in
the article is regenerated from the JSON files in `code/exp/` by the scripts in `code/`.

## What is in this repository

| Path | Contents |
| --- | --- |
| `code/*.py` | Training, evaluation, control, probe, analysis, and figure code (32 scripts) |
| `code/*.sh` | The GPU lane scripts that launched the experiment matrix (7 scripts) |
| `code/exp/<run>/*.json` | Raw per-run evaluation output — 423 files across 63 runs |
| `code/*.log`, `code/*.out` | Unedited stdout of every training and evaluation job |
| `POSS_FREEZE_MANIFEST.md` | Frozen-protocol manifest for the development-independent SemanticPOSS evaluation, with SHA-256 hashes of the five evaluation code files and all 44 checkpoints |

**Trained checkpoints are not included.** The 55 `model.pth` files total 2.6 GB, which
exceeds GitHub's per-file and per-repository limits. Their SHA-256 hashes are recorded in
`POSS_FREEZE_MANIFEST.md`, so any copy can be verified as the exact weights used in the
article. They are available from the corresponding author on request.

No manuscript text, figures, or LaTeX sources are included here.

## Datasets

None of the datasets are redistributed. All four are publicly available from their
original maintainers:

- **SynLiDAR** (source domain) — https://github.com/xiaoaoran/SynLiDAR
- **SemanticKITTI** (target) — http://www.semantic-kitti.org
- **SemanticSTF** (adverse-weather target) — https://github.com/xiaoaoran/SemanticSTF
- **SemanticPOSS** (development-independent target) — http://www.poss.pku.edu.cn/semanticposs.html

Dataset roots are hard-coded near the top of `code/datasets.py` and point at the paths
used on the original machine. Edit them to match your own layout before running anything.
The scripts are published byte-identical to the versions that produced the results, so
these paths were deliberately left unmodified rather than parameterized after the fact.

## Environment

Python 3, PyTorch with CUDA, and `spconv` (the sparse-convolution backend for the
MinkUNet in `code/model.py`), plus `numpy`, `scipy`, and `matplotlib`. Every reported run
used a single NVIDIA GPU; the two GPUs of the workstation ran independent lanes, one process
per GPU (see the `lane_*.sh` scripts for the exact launch commands, batch sizes, and seeds).
`train.py` also supports distributed data-parallel training via `torchrun`, but no reported
run used it.

## Reproducing the analysis without a GPU

Every headline statistic is recomputed from the committed JSON files — no retraining and
no GPU required:

```bash
cd code
python stats_scirep.py     # main table, seed-paired diffs, paired t-tests, 95% CIs
python poss_analysis.py    # development-independent SemanticPOSS results (frozen protocol)
python correlation.py      # sharpness-vs-transfer correlation analysis
python make_fig_sharp2.py  # Figure 2: probe sharpness vs transfer (writes ../paper/srep/figures/)
python make_si_figures_r1.py  # Supplementary Figs S1-S2 (writes ../paper/srep/figures/)
python make_figures.py     # earlier development figures (writes to ../paper/figures/, create it first)
```

`stats_scirep.py` writes `exp/stats_scirep.json` and prints a markdown summary;
`poss_analysis.py` reads `exp/poss_summary.json`. Both recompute from the per-run
`eval_all.json` files rather than from any cached summary.

## Reproducing the experiments

`train.py` is the single entry point; the method under study is selected by flag. All
runs in the article share one base configuration, and the arms differ only in the flags
appended to it:

```bash
BASE="--mode none --stride 10 --warmup_iters 500 --eval_every 12 --eval_max_scans 600 \
      --num_points 80000 --lr 0.1 --epochs 48 --cr 0.5 --batch_size 4"

# source-only baseline
python train.py --seed 2 $BASE --save_path exp/none_s2

# sharpness-aware minimization (the studied intervention)
python train.py --sam_rho 0.05 --seed 2 $BASE --save_path exp/sam005_s2

# PolarMix augmentation baseline, and the combination
python train.py --polarmix --seed 2 $BASE --save_path exp/polarmix_s2
python train.py --sam_rho 0.05 --polarmix --seed 2 $BASE --save_path exp/sampolar_s2
```

Other baselines exposed by the same script: `--pointdr` (PointDR-inspired), `--lasermix`,
`--weather` (UniMix-inspired weather simulation), `--swa_start` (SWA weight averaging),
`--consistency` (sensing-aware consistency), and `--dr_aug`. The capacity check replaces
`--cr 0.5 --batch_size 4` with `--cr 1.0 --batch_size 3` (the full-width `cr1_*` runs). The
2x-compute control is a single source-only run with `--epochs 96 --grad_clip 10`
(`none_long_s1`), which matches SAM's number of gradient evaluations; it is the only run
that needed gradient clipping. The `lane_*.sh` scripts reproduce the
exact run order, and are idempotent — a run whose `eval_all.json` already exists is
skipped, so an interrupted lane can simply be relaunched.

Evaluation and controls (`--cr` must match the value the checkpoint was trained with):

```bash
python eval.py       --ckpt exp/sam005_s2/model.pth --cr 0.5   # KITTI / STF evaluation
python sharpness.py  --ckpt exp/sam005_s2/model.pth --cr 0.5 --rho 0.05 --n_dirs 5 --n_batches 6
python bn_recalib.py     # BatchNorm recalibration control on the unaugmented source stream
python eval_poss.py      # development-independent SemanticPOSS evaluation (frozen protocol)
```

## Development-independent SemanticPOSS evaluation

`POSS_FREEZE_MANIFEST.md` fixes the evaluation split, the 11-class label mapping, the
metric, the checkpoint list, and the analysis plan, and it was written before this study
first read any SemanticPOSS label content. The manifest carries a post-run changelog that
records a provenance correction: SemanticPOSS was concurrently in use in the same
workspace for an unrelated project, so the evaluation is characterized as
*development-independent external evaluation*, not as an author-untouched sealed
confirmation. It also marks which contrasts were prespecified and which are post hoc.
Read the changelog at the bottom of the manifest, not only the frozen body above it.

The five evaluation scripts are hash-pinned. Verify that what you downloaded is what the
manifest describes:

```bash
cd code && sha256sum eval_poss.py bn_recalib.py datasets.py model.py utils.py
```

## Headline results

Mean mIoU over 3 seeds, recomputed from `code/exp/*/eval_all.json` by `stats_scirep.py`:

| Method | SemanticKITTI | SemanticSTF |
| --- | --- | --- |
| source-only | 27.61 ± 1.67 | 20.90 ± 1.19 |
| SAM | 29.63 ± 0.58 | 21.72 ± 0.93 |
| PolarMix-style | 29.00 ± 0.54 | 23.63 ± 0.89 |
| SAM + PolarMix-style | 30.38 ± 0.65 | 24.95 ± 0.61 |

Seed-paired differences with two-sided paired *t*-tests (n = 3):

| Contrast | Δ mIoU | 95% CI | p |
| --- | --- | --- | --- |
| SAM − source-only (KITTI) | +2.02 | [−0.88, 4.92] | 0.096 |
| SAM − source-only (STF) | +0.82 | [−0.93, 2.57] | 0.181 |
| SAM + PolarMix-style − PolarMix-style (KITTI) | +1.38 | [−0.44, 3.20] | 0.082 |
| SAM + PolarMix-style − PolarMix-style (STF) | +1.32 | [−2.10, 4.74] | 0.238 |

With three seeds these intervals include zero; the article reports them as such rather
than claiming significance. On the development-independent SemanticPOSS evaluation under
the frozen protocol, the prespecified SAM − source-only contrast is **+2.61** mIoU (per-seed
+2.71 / +4.22 / +0.90; 95% CI [−1.52, 6.74]) and the post hoc full-width `cr1` pair gives
+2.10, while the post hoc PolarMix-style − source-only contrast is **−0.68** — a negative
result that is reported as-is. All 44 checkpoints appear in the supplementary
table; `exp/poss_summary.json` holds the raw as-is and BN-recalibrated values for each.

## License

MIT — see `LICENSE`. The datasets remain under their own licenses.

## Versions

- **v1.0** (August 2026): the archive as cited in the original submission.
- **v1.1.0** (October 2026): accompanies the revised manuscript. Adds
  `code/make_si_figures_r1.py` and revises `code/make_fig_sharp2.py` (larger text, legend
  below the axes, colour-vision-safe encoding); README corrections (single-GPU runs; the
  capacity check and the 2x-compute control described separately; terminology aligned with
  the manuscript). Training, evaluation, and the five hash-pinned evaluation files are
  unchanged, so `POSS_FREEZE_MANIFEST.md` remains valid.

## Citation

Archived on Zenodo (the DOI is listed on the release page). BibTeX for the article will be
added here once it has a DOI.
