"""Standalone full evaluation of a trained checkpoint on a target domain.
Reports point-level mIoU + per-class IoU, and the feature-nuisance coupling
diagnostic (mean over scans of nuisance_corr at the stem) on the target."""
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import make_target, collate
from model import MinkUNet
from whitening import nuisance_corr
from utils import IoUMeter, NUM_CLASSES


@torch.no_grad()
def run(ckpt, target, cr, voxel_size, use_intensity, device, diag_scans=80, out=None):
    model = MinkUNet(in_channels=4, num_classes=NUM_CLASSES, cr=cr).to(device).eval()
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    ds = make_target(target, voxel_size=voxel_size, use_intensity=use_intensity)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate)
    meter = IoUMeter()
    diag = []
    for i, batch in enumerate(loader):
        feats = batch["feats"].to(device); coords = batch["coords"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(feats, coords, batch["batch_size"])
        vpred = logits.float().argmax(1).cpu().numpy()
        ppred = vpred[batch["invs"][0].numpy()]
        meter.update(ppred, batch["plabels"][0].numpy())
        if i < diag_scans:
            stem = model.capture["stem"].float()
            vb = batch["vbatch"].to(device); vn = batch["vnuis"].to(device)
            diag.append(nuisance_corr(stem, vb, vn))
    res = meter.summary()
    res["target"] = target
    res["nuisance_corr_target"] = float(np.mean(diag)) if diag else None
    res["ckpt"] = ckpt
    if out:
        json.dump(res, open(out, "w"), indent=2)
    print(f"[{target}] mIoU={res['mIoU']*100:.2f}  nuis_corr={res['nuisance_corr_target']}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--targets", nargs="+", default=["kitti", "stf"])
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--no_intensity", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    out_dir = args.out_dir or os.path.dirname(args.ckpt)
    allres = {}
    for t in args.targets:
        allres[t] = run(args.ckpt, t, args.cr, args.voxel_size, not args.no_intensity,
                        device, out=os.path.join(out_dir, f"eval_{t}.json"))
    json.dump(allres, open(os.path.join(out_dir, "eval_all.json"), "w"), indent=2)
