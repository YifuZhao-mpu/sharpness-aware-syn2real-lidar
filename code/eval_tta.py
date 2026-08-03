"""Test-time augmentation (TTA) evaluation: average softmax over dihedral views
(z-rotations x optional x-flip), accumulated in POINT space, then argmax. A standard,
target-label-free inference-time technique, ORTHOGONAL to the training-time method;
reported as an enhancement applied identically to all models. Point-level mIoU."""
import os, json, argparse
import numpy as np
import torch
from datasets import make_target, voxelize
from model import MinkUNet
from utils import IoUMeter, NUM_CLASSES


def views(n_rot):
    V = []
    for k in range(n_rot):
        ang = 2 * np.pi * k / n_rot
        for flip in (False, True):
            V.append((ang, flip))
    return V


@torch.no_grad()
def tta_eval(ckpt, target, cr, voxel_size, device, n_rot=4, max_scans=None, out=None):
    model = MinkUNet(in_channels=4, num_classes=NUM_CLASSES, cr=cr).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    ds = make_target(target, voxel_size=voxel_size)
    use_int = ds.use_intensity
    meter = IoUMeter()
    V = views(n_rot)
    N = len(ds.files) if not max_scans else min(max_scans, len(ds.files))
    for i in range(N):
        coord, inten, label = ds._load(ds.files[i])
        if not use_int:
            inten = np.zeros_like(inten)
        acc = np.zeros((coord.shape[0], NUM_CLASSES), dtype=np.float32)
        for ang, flip in V:
            cs, sn = np.cos(ang), np.sin(ang)
            R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float32)
            c = coord @ R.T
            if flip:
                c = c.copy(); c[:, 0] = -c[:, 0]
            feat_pts = np.concatenate([c, inten], 1).astype(np.float32)
            vcoord, vfeat, _, inv, idx, counts = voxelize(c.astype(np.float32), feat_pts, None, voxel_size)
            n = vcoord.shape[0]
            coords_t = torch.cat([torch.zeros(n, 1, dtype=torch.int32),
                                  torch.from_numpy(vcoord.astype(np.int32))], 1).to(device)
            feats_t = torch.from_numpy(vfeat).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(feats_t, coords_t, 1)
            acc += logits.float().softmax(1).cpu().numpy()[inv]
        meter.update(acc.argmax(1), label)
    res = meter.summary(); res["target"] = target; res["n_views"] = len(V); res["ckpt"] = ckpt
    if out:
        json.dump(res, open(out, "w"), indent=2)
    print(f"[TTA {target}] mIoU={res['mIoU']*100:.2f}  ({len(V)} views, {N} scans)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--targets", nargs="+", default=["kitti", "stf"])
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--n_rot", type=int, default=4)
    ap.add_argument("--max_scans", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    out_dir = os.path.dirname(args.ckpt)
    allres = {}
    for t in args.targets:
        allres[t] = tta_eval(args.ckpt, t, args.cr, args.voxel_size, device, args.n_rot,
                             args.max_scans or None, out=os.path.join(out_dir, f"tta_{t}.json"))
    json.dump(allres, open(os.path.join(out_dir, "tta_all.json"), "w"), indent=2)
