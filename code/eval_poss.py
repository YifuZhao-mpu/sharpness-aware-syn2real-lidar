"""SEALED SemanticPOSS evaluation (round-9 review, P1 response).

SemanticPOSS was NEVER accessed during this project's development (verified: zero
references in code/logs/docs before 2026-07-16). This script implements the frozen
protocol of POSS_FREEZE_MANIFEST.md and is hashed into that manifest BEFORE the
first label is read. One evaluation pass; results reported regardless of outcome;
any bug found after label access is fixed and disclosed in the manifest changelog.

Protocol (fixed from documentation only):
  - Eval split: sequence 03 (the official SemanticPOSS test split), all 500 scans.
  - Common 11-class label space (documentation-derived subsumption mapping):
      0 person, 1 rider, 2 car, 3 trunk, 4 plants, 5 traffic-sign, 6 pole,
      7 building, 8 fence, 9 bike, 10 ground
    POSS raw tags {14 trashcan, 16 cone/stone, 0 unlabeled} -> ignore (255).
    Model predictions (19-class SemanticKITTI space) -> common-11 by subsumption:
      car/truck/other-vehicle -> car; bicycle/motorcycle -> bike;
      bicyclist/motorcyclist -> rider; road/parking/sidewalk/other-ground/terrain
      -> ground; person -> person; building/fence/vegetation/trunk/pole/
      traffic-sign -> their POSS counterparts (plants = vegetation).
  - Metric: point-level mIoU over the 11 common classes (same IoUMeter, batch 1,
    voxel 0.05, final-epoch frozen checkpoints, no TTA, no retuning).
  - Intensity scale: single permitted pre-eval data peek (declared in manifest) at
    the intensity channel range of scan 000000 -> /255 if values exceed 1.5.
  - Secondary (sensitivity): the same checkpoints after common BN recalibration on
    the fixed unaugmented source stream of bn_recalib.py.
"""
import os, json, glob, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import LidarSeg, collate
from utils import build_lut, IoUMeter, NUM_CLASSES
from model import MinkUNet
from bn_recalib import calib_loader, recalibrate

CODE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(CODE, "exp")
POSS_SEQ = "/home/zyf/桌面/project/data/SemanticPOSS/dataset/sequences/03"

COMMON_NAMES = ["person", "rider", "car", "trunk", "plants", "traffic-sign",
                "pole", "building", "fence", "bike", "ground"]
POSS_RAW2COMMON = {0: 255, 4: 0, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 5,
                   12: 5, 13: 6, 14: 255, 15: 7, 16: 255, 17: 8, 21: 9, 22: 10}
POSS_LUT = build_lut(POSS_RAW2COMMON)
#                       car bic mot tru oth per bcl mcl roa par sid oth bui fen veg tru ter pol sig
PRED19_TO_COMMON = np.array([2, 9, 9, 2, 2, 0, 1, 1, 10, 10, 10, 10, 7, 8, 4, 3, 10, 6, 5])

EXCLUDE_PREFIXES = ("pilot_", "_smoke")
EXCLUDE_SUBSTR = (".broken",)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def all_runs():
    runs = []
    for d in sorted(glob.glob(os.path.join(EXP, "*"))):
        n = os.path.basename(d)
        if not os.path.isdir(d):
            continue
        if n.startswith(EXCLUDE_PREFIXES) or any(s in n for s in EXCLUDE_SUBSTR):
            continue
        if os.path.exists(os.path.join(d, "model.pth")) and os.path.exists(os.path.join(d, "args.json")):
            runs.append(n)
    return runs


def intensity_scale():
    """Single permitted pre-eval peek (declared in POSS_FREEZE_MANIFEST.md):
    intensity range of the first scan's 4th column decides /255 vs raw."""
    f = sorted(glob.glob(os.path.join(POSS_SEQ, "velodyne", "*.bin")))[0]
    raw = np.fromfile(f, dtype=np.float32)
    lab = np.fromfile(f.replace("velodyne", "labels").replace(".bin", ".label"), dtype=np.int32)
    ncols = raw.shape[0] // lab.shape[0]
    inten = raw.reshape(-1, ncols)[:, 3]
    mx = float(inten.max())
    return (1.0 / 255.0 if mx > 1.5 else 1.0), {"file": os.path.basename(f), "ncols": ncols,
                                                "inten_min": float(inten.min()), "inten_max": mx}


@torch.no_grad()
def eval_poss(model, ds, device):
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate)
    meter = IoUMeter(num_classes=len(COMMON_NAMES))
    for batch in loader:
        feats = batch["feats"].to(device); coords = batch["coords"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(feats, coords, batch["batch_size"])
        vpred19 = logits.float().argmax(1).cpu().numpy()
        ppred = PRED19_TO_COMMON[vpred19[batch["invs"][0].numpy()]]
        meter.update(ppred, batch["plabels"][0].numpy())
    iou = meter.iou_per_class()
    return {"mIoU": float(np.nanmean(iou)),
            "per_class": {COMMON_NAMES[i]: (None if np.isnan(iou[i]) else round(float(iou[i]) * 100, 2))
                          for i in range(len(COMMON_NAMES))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--skip_bn_recalib", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda", 0)

    runs = args.runs or all_runs()
    scale, peek = intensity_scale()
    log(f"intensity peek {peek} -> scale {scale}")
    files = sorted(glob.glob(os.path.join(POSS_SEQ, "velodyne", "*.bin")))
    ds = LidarSeg(files, POSS_LUT, voxel_size=args.voxel_size, num_points=10**9,
                  train=False, use_intensity=True, intensity_scale=scale)
    log(f"SemanticPOSS seq03: {len(files)} scans; {len(runs)} checkpoints")

    cal_files, cal_loader_ = None, None
    summary = {}
    for run in runs:
        cr = json.load(open(os.path.join(EXP, run, "args.json")))["cr"]
        model = MinkUNet(in_channels=4, num_classes=NUM_CLASSES, cr=cr).to(device).eval()
        model.load_state_dict(torch.load(os.path.join(EXP, run, "model.pth"), map_location=device))
        res = {"run": run, "cr": cr, "n_scans": len(files),
               "intensity_scale": scale, "intensity_peek": peek,
               "asis": eval_poss(model, ds, device)}
        line = f"{run}: POSS mIoU {res['asis']['mIoU']*100:.2f}"
        if not args.skip_bn_recalib:
            if cal_loader_ is None:
                cal_files, cal_loader_ = calib_loader(247, 4, args.voxel_size)
            recalibrate(model, cal_loader_, device)
            res["bn_recalib"] = eval_poss(model, ds, device)
            line += f" | BN-recal {res['bn_recalib']['mIoU']*100:.2f}"
        json.dump(res, open(os.path.join(EXP, run, "eval_poss.json"), "w"), indent=2)
        summary[run] = {"asis": res["asis"]["mIoU"] * 100,
                        "bn_recalib": (res.get("bn_recalib", {}).get("mIoU", np.nan) * 100
                                       if "bn_recalib" in res else None)}
        log(line)
    json.dump(summary, open(os.path.join(EXP, "poss_summary.json"), "w"), indent=2)
    log("sealed evaluation complete")


if __name__ == "__main__":
    main()
