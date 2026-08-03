"""BN recalibration test (round-9 review, P2 response).

SAM training updates BatchNorm running statistics on BOTH forward passes (no freeze
on the perturbation pass), so SAM checkpoints reach inference with different BN
buffers than source-only checkpoints. This script tests whether that inference-time
BN-buffer difference explains the target-domain gains: for each core checkpoint we
reset the BN running stats and recompute them on ONE FIXED, UNAUGMENTED source
calibration stream (identical scans, identical order, for every checkpoint), then
re-evaluate both targets. If the SAM/composition advantage survives common
recalibration, the BN confound cannot account for it; if it vanishes, BN-statistics
dynamics -- not the SAM objective -- drove the result.

Recipe matches the SWA branch of train.py: reset_running_stats() + momentum=None
(cumulative moving average), train-mode no-grad forwards, then eval.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import LidarSeg, synlidar_files, make_target, collate
from utils import SYNLIDAR_LUT, IoUMeter, NUM_CLASSES
from model import MinkUNet

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")

CORE_RUNS = ["none_full", "none_s2", "none_s3",
             "sam005_full", "sam005_s2", "sam005_s3",
             "polarmix_s1", "polarmix_s2", "polarmix_s3",
             "sampolar_s1", "sampolar_s2", "sampolar_s3"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calib_loader(stride, batch_size, voxel_size, num_workers=4):
    """Fixed unaugmented source calibration stream: every `stride`-th SynLiDAR scan of
    the FULL sorted file list (not the training subset), train=False -> deterministic,
    no augmentation, no point subsampling. Identical for every checkpoint."""
    files = synlidar_files(stride=1)[::stride]
    ds = LidarSeg(files, SYNLIDAR_LUT, voxel_size=voxel_size, num_points=10**9,
                  train=False, use_intensity=True)
    return files, DataLoader(ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate)


@torch.no_grad()
def evaluate(model, target, voxel_size, device, max_scans=None):
    """Point-level mIoU on a target val set; identical code path to eval.py."""
    ds = make_target(target, voxel_size=voxel_size, use_intensity=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate)
    meter = IoUMeter()
    for i, batch in enumerate(loader):
        if max_scans and i >= max_scans:
            break
        feats = batch["feats"].to(device); coords = batch["coords"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(feats, coords, batch["batch_size"])
        vpred = logits.float().argmax(1).cpu().numpy()
        ppred = vpred[batch["invs"][0].numpy()]
        meter.update(ppred, batch["plabels"][0].numpy())
    return meter.summary()


@torch.no_grad()
def recalibrate(model, loader, device):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats(); m.momentum = None  # cumulative moving average
    model.train()
    nb = 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(batch["feats"].to(device), batch["coords"].to(device), batch["batch_size"])
        nb += 1
    model.eval()
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=CORE_RUNS)
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--calib_stride", type=int, default=247)   # 198396 scans -> 804
    ap.add_argument("--calib_bs", type=int, default=4)
    ap.add_argument("--max_eval_scans", type=int, default=0)   # smoke tests only
    ap.add_argument("--verify_before", action="store_true",
                    help="also re-evaluate WITHOUT recalibration to confirm this "
                         "eval loop reproduces the stored eval_all.json numbers")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    mx = args.max_eval_scans or None

    files, loader = calib_loader(args.calib_stride, args.calib_bs, args.voxel_size)
    log(f"calibration stream: {len(files)} scans (stride {args.calib_stride}), "
        f"bs {args.calib_bs}, unaugmented, deterministic")

    for run in args.runs:
        ckpt = os.path.join(EXP, run, "model.pth")
        before_path = os.path.join(EXP, run, "eval_all.json")
        before = json.load(open(before_path))
        model = MinkUNet(in_channels=4, num_classes=NUM_CLASSES, cr=args.cr).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device))

        res = {"run": run, "ckpt": ckpt,
               "calib": {"n_scans": len(files), "stride": args.calib_stride,
                         "batch_size": args.calib_bs, "unaugmented": True,
                         "recipe": "reset_running_stats+momentum=None (cumulative)"},
               "before": {"kitti": before["kitti"]["mIoU"], "stf": before["stf"]["mIoU"]}}

        if args.verify_before:
            vk = evaluate(model, "kitti", args.voxel_size, device, mx)
            vs = evaluate(model, "stf", args.voxel_size, device, mx)
            res["verify_before"] = {"kitti": vk["mIoU"], "stf": vs["mIoU"]}
            log(f"{run} VERIFY stored K={before['kitti']['mIoU']*100:.2f} S={before['stf']['mIoU']*100:.2f} "
                f"| re-eval K={vk['mIoU']*100:.2f} S={vs['mIoU']*100:.2f}")

        t0 = time.time()
        nb = recalibrate(model, loader, device)
        log(f"{run}: BN recalibrated over {nb} batches in {time.time()-t0:.0f}s")
        k = evaluate(model, "kitti", args.voxel_size, device, mx)
        s = evaluate(model, "stf", args.voxel_size, device, mx)
        res["after"] = {"kitti": k["mIoU"], "stf": s["mIoU"],
                        "kitti_per_class": k.get("per_class"), "stf_per_class": s.get("per_class")}
        out = os.path.join(EXP, run, "bn_recalib.json")
        json.dump(res, open(out, "w"), indent=2)
        log(f"{run}: K {before['kitti']['mIoU']*100:.2f} -> {k['mIoU']*100:.2f} "
            f"({(k['mIoU']-before['kitti']['mIoU'])*100:+.2f}) | "
            f"S {before['stf']['mIoU']*100:.2f} -> {s['mIoU']*100:.2f} "
            f"({(s['mIoU']-before['stf']['mIoU'])*100:+.2f})  -> {out}")


if __name__ == "__main__":
    main()
