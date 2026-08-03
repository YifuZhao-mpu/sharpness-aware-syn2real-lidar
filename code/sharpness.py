"""Loss-landscape sharpness of a trained checkpoint (mechanism evidence for the
flat-minima story). Filter-normalized random perturbations (Li et al. 2018):
for each weight tensor p, sample d~N(0,1), rescale to rho*||p|| (filter Frobenius),
measure mean source-domain CE increase L(w+eps)-L(w). Lower = flatter minimum.
Compare source-only (expected sharp) vs SAM (expected flat)."""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import make_source, collate
from model import MinkUNet
from utils import NUM_CLASSES, IGNORE


@torch.no_grad()
def eval_loss(model, batches, device):
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    tot, n = 0.0, 0
    for b in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["feats"].to(device), b["coords"].to(device), b["batch_size"])
            loss = ce(logits, b["vlabels"].to(device))
        tot += float(loss); n += 1
    return tot / max(1, n)


@torch.no_grad()
def filter_perturb(model, rho, gen):
    """Apply a filter-normalized random perturbation of relative size rho; return the deltas."""
    deltas = []
    for p in model.parameters():
        if p.dim() < 1:
            deltas.append(None); continue
        d = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
        d = d / (d.norm() + 1e-12) * (p.norm() + 1e-12) * rho   # scale to rho*||p||
        p.add_(d); deltas.append(d)
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.05)     # relative perturbation size
    ap.add_argument("--n_dirs", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=12)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    model = MinkUNet(in_channels=4, num_classes=NUM_CLASSES, cr=args.cr).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    ds = make_source(stride=args.stride)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=6, collate_fn=collate)
    batches = []
    for i, b in enumerate(loader):
        if i >= args.n_batches:
            break
        batches.append(b)

    base = eval_loss(model, batches, device)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    incs = []
    for _ in range(args.n_dirs):
        deltas = filter_perturb(model, args.rho, gen)
        l = eval_loss(model, batches, device)
        with torch.no_grad():
            for p, d in zip(model.parameters(), deltas):
                if d is not None:
                    p.sub_(d)
        incs.append(l - base)
    incs = np.array(incs)
    res = {"ckpt": args.ckpt, "rho": args.rho, "base_loss": base,
           "sharpness_mean": float(incs.mean()), "sharpness_std": float(incs.std()),
           "n_dirs": args.n_dirs}
    print(json.dumps(res, indent=2))
    out = os.path.join(os.path.dirname(args.ckpt), "sharpness.json")
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
