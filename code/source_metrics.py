"""Compute SOURCE-domain metrics (held-out SynLiDAR) for a checkpoint: source CE loss and
source mIoU. These are the ordinary target-free signals a practitioner could use for model
selection (no target data). We later compare them against sharpness as a selection signal."""
import os, json, glob, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import LidarSeg, collate, synlidar_files, DATA_ROOT
from utils import SYNLIDAR_LUT, IoUMeter, NUM_CLASSES, IGNORE
from model import MinkUNet


def heldout_source_files(n=400):
    """Deterministic held-out SynLiDAR subset (stride-200, offset-5: disjoint from the
    stride-10 training subset since offset 5 is never divisible by 10)."""
    files = []
    for s in [f"{i:02d}" for i in range(13)]:
        files += sorted(glob.glob(os.path.join(DATA_ROOT, "SynLiDAR", "sequences", s, "velodyne", "*.bin")))
    files.sort()
    held = files[5::200]
    return held[:n]


@torch.no_grad()
def run(ckpt, cr, device, n_scans=400):
    model = MinkUNet(in_channels=4, num_classes=NUM_CLASSES, cr=cr).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    ds = LidarSeg(heldout_source_files(n_scans), SYNLIDAR_LUT, 0.05, num_points=10**9,
                  train=False, use_intensity=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate)
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    meter = IoUMeter(); tot_loss, nb = 0.0, 0
    for batch in loader:
        feats = batch["feats"].to(device); coords = batch["coords"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(feats, coords, batch["batch_size"])
            loss = ce(logits.float(), batch["vlabels"].to(device))
        tot_loss += float(loss); nb += 1
        vpred = logits.float().argmax(1).cpu().numpy()
        meter.update(vpred[batch["invs"][0].numpy()], batch["plabels"][0].numpy())
    return {"src_loss": tot_loss / max(1, nb), "src_mIoU": meter.miou() * 100}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--n_scans", type=int, default=400)
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    res = run(args.ckpt, args.cr, device, args.n_scans)
    out = os.path.join(os.path.dirname(args.ckpt), "source_metrics.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"{args.ckpt}: src_loss={res['src_loss']:.4f} src_mIoU={res['src_mIoU']:.2f}")
