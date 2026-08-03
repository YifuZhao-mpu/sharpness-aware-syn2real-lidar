"""Rigorous sharpness probe (round-9 review, P4 response). Replaces sharpness.py.

Fixes every measurement defect the round-9 audit identified in the original probe:
  1. TRUE per-filter normalization (Li et al. 2018): each dim-0 slice of every
     weight tensor with dim>=2 is rescaled to rho*||w_i||; BN affine params and
     biases are NOT perturbed (the original probe rescaled per whole tensor and
     perturbed everything).
  2. Fixed HELD-OUT source batches: scans at offset 5 of the stride-10 grid, i.e.
     disjoint from the stride-10 training subset (files[::10]); train=False ->
     unaugmented, no subsampling, deterministic; identical files for every checkpoint.
  3. Model in eval mode -> BN uses stored running stats; no batch-stat noise.
  4. 32 directions (was 5), shared across checkpoints (same seed per direction
     index + identical shapes -> paired comparisons).
  5. Full forensic record: per-direction losses, batch file list, seeds, config.

Writes exp/<run>/sharpness2.json.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import LidarSeg, synlidar_files, collate
from utils import SYNLIDAR_LUT, IoUMeter, NUM_CLASSES, IGNORE
from model import MinkUNet

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")

# FROZEN correlation cohort (round-9 reviewer-endorsed rule, fixed BEFORE measuring):
# every non-pilot, non-broken, half-width final checkpoint from the flatness-
# intervention families {source-only, vanilla same-view SAM, SWA}, all seeds, radii,
# and schedules; every loss-bearing forward uses only the shared base source view.
FROZEN_COHORT = ["none_full", "none_s2", "none_s3", "none_long_s1",
                 "sam001_s1", "sam005_full", "sam005_s2", "sam005_s3",
                 "sam010_s1", "sam020_s1", "swa_s1", "swa_s2", "swa_s3"]
# measured with the same probe but OUTSIDE the correlation (different training data
# distribution or corrupted-view losses); shown separately, as in Fig. 4:
EXTRA = ["samsens_s1", "samsens_s2", "dr_full_s1", "neg_whiten_main",
         "polarmix_s1", "polarmix_s2", "polarmix_s3",
         "sampolar_s1", "sampolar_s2", "sampolar_s3",
         "pointdr_s1", "pointdr_s2", "pointdr_s3"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def heldout_batches(voxel_size, n_scans=48, batch_size=4):
    """Fixed held-out scans: offset-5 stride-10 grid (disjoint from the training
    subset files[::10]), evenly spaced, deterministic."""
    allf = synlidar_files(stride=1)
    held = allf[5::10]
    files = held[::max(1, len(held) // n_scans)][:n_scans]
    ds = LidarSeg(files, SYNLIDAR_LUT, voxel_size=voxel_size, num_points=10**9,
                  train=False, use_intensity=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        collate_fn=collate)
    return files, [b for b in loader]   # materialize once, reuse for all checkpoints


@torch.no_grad()
def mean_loss(model, batches, device, ce):
    tot = 0.0
    for b in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["feats"].to(device), b["coords"].to(device), b["batch_size"])
            loss = ce(logits, b["vlabels"].to(device))
        tot += float(loss)
    return tot / len(batches)


def unit_direction(shapes_params, seed):
    """One random direction per weight tensor (dim>=2 only), from a fixed CPU seed
    -> identical unit direction across checkpoints with the same architecture."""
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(p.shape, generator=gen) if p.dim() >= 2 else None
            for p in shapes_params]


@torch.no_grad()
def filter_scaled(direction, params, rho):
    """Per-filter scaling: each dim-0 slice of d rescaled to rho*||w_slice||."""
    out = []
    for d, p in zip(direction, params):
        if d is None:
            out.append(None); continue
        d = d.to(p.device, p.dtype)
        dm = d.reshape(d.shape[0], -1); pm = p.reshape(p.shape[0], -1)
        scale = rho * pm.norm(dim=1) / (dm.norm(dim=1) + 1e-12)
        out.append((dm * scale[:, None]).reshape(p.shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=FROZEN_COHORT + EXTRA)
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--n_dirs", type=int, default=32)
    ap.add_argument("--n_scans", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dir_seed_base", type=int, default=1000)
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    files, batches = heldout_batches(args.voxel_size, args.n_scans, args.batch_size)
    log(f"held-out probe set: {len(files)} scans (offset-5 stride-10 grid, disjoint "
        f"from training subset), {len(batches)} batches of {args.batch_size}, eval-mode")

    for run in args.runs:
        ckpt = os.path.join(EXP, run, "model.pth")
        if not os.path.exists(ckpt):
            log(f"{run}: MISSING checkpoint, skipped"); continue
        model = MinkUNet(in_channels=4, num_classes=NUM_CLASSES, cr=args.cr).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        params = [p for p in model.parameters()]

        base = mean_loss(model, batches, device, ce)
        incs = []
        t0 = time.time()
        for di in range(args.n_dirs):
            d = unit_direction(params, args.dir_seed_base + di)
            delta = filter_scaled(d, params, args.rho)
            with torch.no_grad():
                for p, dl in zip(params, delta):
                    if dl is not None:
                        p.add_(dl)
            lp = mean_loss(model, batches, device, ce)
            with torch.no_grad():
                for p, dl in zip(params, delta):
                    if dl is not None:
                        p.sub_(dl)
            incs.append(lp - base)
        incs = np.array(incs)
        res = {"run": run, "ckpt": ckpt, "base_loss": base,
               "sharpness_mean": float(incs.mean()),
               "sharpness_std": float(incs.std(ddof=1)),
               "sharpness_median": float(np.median(incs)),
               "per_direction": [float(x) for x in incs],
               "config": {"rho": args.rho, "n_dirs": args.n_dirs,
                          "normalization": "per-filter (dim-0 slices of dim>=2 weights); "
                                           "BN affine and biases not perturbed",
                          "model_mode": "eval (BN running stats)",
                          "data": "held-out unaugmented SynLiDAR, fixed",
                          "dir_seed_base": args.dir_seed_base,
                          "batch_files": files,
                          "in_frozen_cohort": run in FROZEN_COHORT}}
        json.dump(res, open(os.path.join(EXP, run, "sharpness2.json"), "w"), indent=2)
        log(f"{run}: base={base:.4f} sharp={incs.mean():.4f}±{incs.std(ddof=1):.4f} "
            f"(median {np.median(incs):.4f}) [{time.time()-t0:.0f}s]")

    # correlation over the FROZEN cohort only (extras reported separately)
    rows = []
    for run in FROZEN_COHORT:
        sp = os.path.join(EXP, run, "sharpness2.json")
        ep = os.path.join(EXP, run, "eval_all.json")
        if os.path.exists(sp) and os.path.exists(ep):
            e = json.load(open(ep))
            rows.append((run, json.load(open(sp))["sharpness_mean"],
                         (e["kitti"]["mIoU"] + e["stf"]["mIoU"]) / 2 * 100,
                         e["kitti"]["mIoU"] * 100, e["stf"]["mIoU"] * 100))
    if len(rows) >= 3:
        sh = np.array([r[1] for r in rows]); mm = np.array([r[2] for r in rows])
        kk = np.array([r[3] for r in rows]); ss = np.array([r[4] for r in rows])
        def sp_r(x, y):
            rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
            return float(np.corrcoef(rx, ry)[0, 1])
        stats = {"cohort": [r[0] for r in rows], "n": len(rows),
                 "pearson_mean": float(np.corrcoef(sh, mm)[0, 1]),
                 "pearson_kitti": float(np.corrcoef(sh, kk)[0, 1]),
                 "pearson_stf": float(np.corrcoef(sh, ss)[0, 1]),
                 "spearman_mean": sp_r(sh, mm),
                 "pearson_log_mean": float(np.corrcoef(np.log(np.maximum(sh, 1e-6)), mm)[0, 1]),
                 "rows": [{"name": r[0], "sharpness2": r[1], "mean_mIoU": r[2],
                           "kitti": r[3], "stf": r[4]} for r in rows]}
        json.dump(stats, open(os.path.join(EXP, "corr_stats2.json"), "w"), indent=2)
        log(f"FROZEN cohort n={stats['n']}: Pearson(mean)={stats['pearson_mean']:.3f} "
            f"K={stats['pearson_kitti']:.3f} S={stats['pearson_stf']:.3f} "
            f"Spearman={stats['spearman_mean']:.3f} log={stats['pearson_log_mean']:.3f}")


if __name__ == "__main__":
    main()
