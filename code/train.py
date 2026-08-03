"""Train MinkUNet on SynLiDAR (single-source) for synthetic->real DG.
Supports source-only and Sensing-Style Whitening (SSW). DDP (torchrun) or single-GPU.

Usage (2-GPU DDP):
  torchrun --nproc_per_node=2 train.py --mode sensing --lambda_w 0.1 --save_path exp/ssw
"""
import os, json, math, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import make_source, make_target, collate
from model import MinkUNet
from whitening import whitening_loss, nuisance_corr
from utils import IoUMeter, NUM_CLASSES, IGNORE


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def log(msg):
    if get_rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_sensing_view(feats, coords, use_intensity, drop=0.3):
    """Range-biased ray-drop + intensity jitter -> a 'sim->real sensing' perturbed view.
    Returns (keep_mask, perturbed_feats, perturbed_coords)."""
    M = feats.shape[0]
    rng = feats[:, :3].norm(dim=1)
    rn = rng / (rng.max() + 1e-6)
    pdrop = (drop * (0.5 + rn)).clamp(0.05, 0.9)
    keep = torch.rand(M, device=feats.device) > pdrop
    if keep.sum() < max(1000, 0.2 * M):
        keep = torch.rand(M, device=feats.device) > drop
    fp = feats[keep].clone()
    if use_intensity:
        s = torch.empty((), device=fp.device).uniform_(0.5, 1.5)
        fp[:, 3] = fp[:, 3] * s + torch.randn_like(fp[:, 3]) * 0.05
    return keep, fp, coords[keep].contiguous()


def make_strong_view(feats, coords, vlabels, vbatch, use_intensity):
    """PointDR strong/augmented view: random voxel dropout (keep 80-100%) + injected noise
    voxels (labelled ignore). Operates on the voxelized batch. Returns strong tensors."""
    M = feats.shape[0]
    keep = torch.rand(M, device=feats.device) > torch.empty((), device=feats.device).uniform_(0.0, 0.2)
    f, c, l, b = feats[keep], coords[keep], vlabels[keep], vbatch[keep]
    # injected noise: a few hundred random voxels per scan, label = ignore (255)
    B = int(vbatch.max().item()) + 1
    noise_chunks_f, noise_chunks_c, noise_chunks_l, noise_chunks_b = [f], [c], [l], [b]
    for bi in range(B):
        cb = coords[vbatch == bi]
        if cb.shape[0] < 10:
            continue
        nn_ = int(torch.randint(0, 600, (1,)).item())
        if nn_ == 0:
            continue
        lo = cb[:, 1:].min(0).values.float(); hi = cb[:, 1:].max(0).values.float()
        rc = (lo + torch.rand(nn_, 3, device=feats.device) * (hi - lo).clamp_min(1)).round().int()
        nc = torch.cat([torch.full((nn_, 1), bi, device=feats.device, dtype=torch.int32), rc], 1)
        nf = torch.randn(nn_, feats.shape[1], device=feats.device) * 0.1
        if not use_intensity:
            nf[:, 3] = 0
        noise_chunks_f.append(nf); noise_chunks_c.append(nc)
        noise_chunks_l.append(torch.full((nn_,), IGNORE, device=feats.device, dtype=l.dtype))
        noise_chunks_b.append(torch.full((nn_,), bi, device=feats.device, dtype=b.dtype))
    return (torch.cat(noise_chunks_f), torch.cat(noise_chunks_c),
            torch.cat(noise_chunks_l), torch.cat(noise_chunks_b))


def pointdr_loss(feat_w, vlabels_w, feat_s, vlabels_s, memo_bank, num_classes, T, m, init):
    """PointDR prototype-InfoNCE (Xiao et al., CVPR'23): build per-class prototypes from the
    weak view, momentum-update a memory bank, classify strong-view point features against it."""
    fw = torch.nn.functional.normalize(feat_w.float(), dim=1)
    fs = torch.nn.functional.normalize(feat_s.float(), dim=1)
    proto = torch.zeros(num_classes, fw.shape[1], device=fw.device)
    for c in range(num_classes):
        mask = vlabels_w == c
        if mask.any():
            proto[c] = fw[mask].mean(0)
    proto = torch.nn.functional.normalize(proto + 1e-8, dim=1)
    with torch.no_grad():
        new_bank = proto if init else (memo_bank * m + proto * (1 - m))
    logits = (fs @ new_bank.t().detach()) / T
    loss = torch.nn.functional.cross_entropy(logits, vlabels_s, ignore_index=IGNORE)
    return loss, new_bank.detach()


def sensing_consistency(model, feats, coords, bs, logits_clean, vbatch, use_intensity,
                        drop=0.3, cons_mode="sensing", conf=0.0):
    """Consistency under a perturbed view; enforce perturbed predictions match the clean
    (stop-grad) predictions on co-visible voxels.
      cons_mode='sensing' (ours): RANGE-BIASED ray-drop + INTENSITY jitter (sim->real sensing model).
      cons_mode='uniform' (DGLSS-style baseline): uniform random subsample, no intensity perturb."""
    M = feats.shape[0]
    if cons_mode == "sensing":
        rng = feats[:, :3].norm(dim=1)
        rn = rng / (rng.max() + 1e-6)
        pdrop = (drop * (0.5 + rn)).clamp(0.05, 0.9)     # farther -> dropped more (real ray-drop)
        keep = torch.rand(M, device=feats.device) > pdrop
    else:  # uniform density consistency (DGLSS-style)
        keep = torch.rand(M, device=feats.device) > drop
    if keep.sum() < max(1000, 0.2 * M):
        keep = torch.rand(M, device=feats.device) > drop
    fp = feats[keep].clone()
    if cons_mode == "sensing" and use_intensity:
        s = torch.empty((), device=fp.device).uniform_(0.5, 1.5)
        fp[:, 3] = fp[:, 3] * s + torch.randn_like(fp[:, 3]) * 0.05
    cp = coords[keep].contiguous()
    logits_p = model(fp, cp, bs)
    tgt = logits_clean[keep].detach().softmax(-1)
    ce_pt = -(tgt * logits_p.log_softmax(-1)).sum(-1)          # per-voxel consistency CE
    if conf > 0:                                               # confidence masking (FixMatch-style)
        mask = tgt.max(-1).values > conf
        if mask.sum() < 1:
            return logits_p.new_zeros(())
        return ce_pt[mask].mean()
    return ce_pt.mean()


@torch.no_grad()
def evaluate(model, name, device, voxel_size, use_intensity, max_scans=None):
    model.eval()
    ds = make_target(name, voxel_size=voxel_size, use_intensity=use_intensity)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate)
    meter = IoUMeter()
    for i, batch in enumerate(loader):
        if max_scans and i >= max_scans:
            break
        feats = batch["feats"].to(device)
        coords = batch["coords"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(feats, coords, batch["batch_size"])
        vpred = logits.float().argmax(1).cpu().numpy()      # [Mtot] voxel preds
        inv = batch["invs"][0].numpy()                       # [N] -> voxel idx
        ppred = vpred[inv]
        meter.update(ppred, batch["plabels"][0].numpy())
    return meter.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="none", choices=["none", "blunt", "random", "sensing"])
    ap.add_argument("--lambda_w", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)   # per GPU
    ap.add_argument("--lr", type=float, default=0.24)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--num_points", type=int, default=80000)
    ap.add_argument("--stride", type=int, default=None)    # SynLiDAR subsample stride
    ap.add_argument("--no_intensity", action="store_true")
    ap.add_argument("--dr_aug", action="store_true")   # PointDR-style domain randomization baseline
    ap.add_argument("--polarmix", action="store_true") # PolarMix strong augmentation baseline
    ap.add_argument("--lasermix", action="store_true") # LaserMix strong augmentation (inclination-axis mixing)
    ap.add_argument("--weather", action="store_true")  # LISA-style adverse-weather simulation (UniMix weather component)
    ap.add_argument("--sam_rho", type=float, default=0.0)   # >0 enables sharpness-aware minimization
    ap.add_argument("--sam_sensing", action="store_true")   # ascent loss on a sensing-perturbed view (ours)
    ap.add_argument("--swa_start", type=int, default=0)     # >0: SWA weight-averaging from this epoch (2nd flat-minima method)
    ap.add_argument("--grad_clip", type=float, default=0.0) # >0: clip grad norm (stability for long schedules)
    ap.add_argument("--pointdr", action="store_true")       # PointDR (Xiao CVPR'23) reimplementation baseline
    ap.add_argument("--pointdr_lambda", type=float, default=0.1)
    ap.add_argument("--pointdr_T", type=float, default=1.0)
    ap.add_argument("--pointdr_m", type=float, default=0.99)
    ap.add_argument("--consistency", type=float, default=0.0)  # >0 enables sensing-aware consistency (SAC)
    ap.add_argument("--cons_drop", type=float, default=0.3)
    ap.add_argument("--cons_mode", default="sensing", choices=["sensing", "uniform"])
    ap.add_argument("--cons_rampup", type=int, default=5)   # epochs to ramp consistency weight 0->1
    ap.add_argument("--cons_conf", type=float, default=0.5)  # confidence threshold for consistency
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--save_path", default="exp/run")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--max_iters", type=int, default=0)    # >0 for smoke test
    ap.add_argument("--eval_max_scans", type=int, default=0)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + get_rank())
    use_intensity = not args.no_intensity

    os.makedirs(args.save_path, exist_ok=True)
    if get_rank() == 0:
        json.dump(vars(args), open(os.path.join(args.save_path, "args.json"), "w"), indent=2)

    src = make_source(voxel_size=args.voxel_size, num_points=args.num_points,
                      stride=args.stride, use_intensity=use_intensity, dr_aug=args.dr_aug,
                      polarmix=args.polarmix, lasermix=args.lasermix, weather=args.weather)
    log(f"SynLiDAR source scans: {len(src)}  mode={args.mode} lambda_w={args.lambda_w} cr={args.cr}")
    sampler = DistributedSampler(src) if is_dist() else None
    loader = DataLoader(src, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=8, collate_fn=collate,
                        drop_last=True, pin_memory=True, persistent_workers=True)

    in_ch = 4
    model = MinkUNet(in_channels=in_ch, num_classes=NUM_CLASSES, cr=args.cr).to(device)
    if is_dist():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    find_unused_parameters=False)
    core = model.module if is_dist() else model

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd, nesterov=True)
    iters_per_epoch = len(loader)
    total_iters = args.epochs * iters_per_epoch if args.max_iters == 0 else args.max_iters

    def lr_at(it):
        if it < args.warmup_iters:
            return args.lr * it / max(1, args.warmup_iters)
        p = (it - args.warmup_iters) / max(1, total_iters - args.warmup_iters)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, p)))

    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    history = []
    git = 0
    swa_state, swa_n = None, 0   # SWA: running average of weights (flat-minima baseline)
    memo_bank = None             # PointDR: momentum memory bank of class prototypes
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        run_ce = run_w = run_nc = 0.0
        nb = 0
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(git)
            feats = batch["feats"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            vlabels = batch["vlabels"].to(device, non_blocking=True)
            vnuis = batch["vnuis"].to(device, non_blocking=True)
            vbatch = batch["vbatch"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.pointdr:
                    logits, feat_w = model(feats, coords, batch["batch_size"], return_feat=True)
                else:
                    logits = model(feats, coords, batch["batch_size"])
                stem_clean = core.capture["stem"]   # snapshot before any 2nd forward overwrites capture
                loss_ce = ce(logits, vlabels)
                if args.mode != "none" and args.lambda_w > 0:
                    loss_w = whitening_loss(stem_clean.float(), vbatch, vnuis, mode=args.mode)
                else:
                    loss_w = logits.new_zeros(())
                loss = loss_ce + args.lambda_w * loss_w
                if args.pointdr:   # PointDR: strong view + prototype-InfoNCE
                    sf, sc, sl, sb = make_strong_view(feats, coords, vlabels, vbatch, use_intensity)
                    _, feat_s = model(sf, sc, batch["batch_size"], return_feat=True)
                    loss_pdr, memo_bank = pointdr_loss(feat_w, vlabels, feat_s, sl, memo_bank,
                                                       NUM_CLASSES, args.pointdr_T, args.pointdr_m,
                                                       init=(memo_bank is None))
                    loss = loss + args.pointdr_lambda * loss_pdr
                    run_w += float(loss_pdr)
                if args.consistency > 0:
                    cons_w = args.consistency * min(1.0, epoch / max(1, args.cons_rampup))  # ramp-up
                    loss_cons = sensing_consistency(model, feats, coords, batch["batch_size"],
                                                    logits, vbatch, use_intensity, args.cons_drop,
                                                    cons_mode=args.cons_mode, conf=args.cons_conf)
                    loss = loss + cons_w * loss_cons
                    run_w += float(loss_cons)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.sam_rho > 0:   # sharpness-aware minimization: ascend then descend
                with torch.no_grad():
                    gn = torch.norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]))
                    scale = args.sam_rho / (gn + 1e-12)
                    ew = []
                    for p in model.parameters():
                        if p.grad is None:
                            ew.append(None); continue
                        e = p.grad * scale; p.add_(e); ew.append(e)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.sam_sensing:   # evaluate sharpness on a sim->real sensing-perturbed view
                        keep_s, fp_s, cp_s = make_sensing_view(feats, coords, use_intensity)
                        logits2 = model(fp_s, cp_s, batch["batch_size"])
                        loss2 = ce(logits2, vlabels[keep_s])
                    else:
                        logits2 = model(feats, coords, batch["batch_size"])
                        loss2 = ce(logits2, vlabels)
                        if args.mode != "none" and args.lambda_w > 0:
                            loss2 = loss2 + args.lambda_w * whitening_loss(
                                core.capture["stem"].float(), vbatch, vnuis, mode=args.mode)
                loss2.backward()
                with torch.no_grad():
                    for p, e in zip(model.parameters(), ew):
                        if e is not None:
                            p.sub_(e)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            run_ce += loss_ce.item(); run_w += float(loss_w); nb += 1
            if nb % 50 == 0:
                with torch.no_grad():
                    run_nc = nuisance_corr(stem_clean.float(), vbatch, vnuis)
            git += 1
            if args.max_iters and git >= args.max_iters:
                break
        log(f"epoch {epoch} done in {time.time()-t0:.0f}s ce={run_ce/max(1,nb):.3f} "
            f"w={run_w/max(1,nb):.4f} nuis_corr={run_nc:.3f} lr={lr_at(git):.4f}")
        if args.max_iters and git >= args.max_iters:
            break
        if args.swa_start > 0 and epoch >= args.swa_start:   # accumulate SWA average (once per epoch)
            with torch.no_grad():
                sd = {k: v.detach().float().clone() for k, v in core.state_dict().items()}
                if swa_state is None:
                    swa_state = sd
                else:
                    for k in swa_state:
                        if swa_state[k].is_floating_point():
                            swa_state[k].mul_(swa_n / (swa_n + 1)).add_(sd[k] / (swa_n + 1))
                        else:
                            swa_state[k] = sd[k]
                swa_n += 1
        if get_rank() == 0 and ((epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1):
            ev_max = args.eval_max_scans or None
            k = evaluate(core, "kitti", device, args.voxel_size, use_intensity, ev_max)
            s = evaluate(core, "stf", device, args.voxel_size, use_intensity, ev_max)
            log(f"  EVAL epoch {epoch}: KITTI mIoU={k['mIoU']*100:.2f}  STF mIoU={s['mIoU']*100:.2f}")
            history.append({"epoch": epoch, "kitti": k, "stf": s})
            json.dump(history, open(os.path.join(args.save_path, "history.json"), "w"), indent=2)
            torch.save(core.state_dict(), os.path.join(args.save_path, "model.pth"))
        if is_dist():
            dist.barrier()

    if args.swa_start > 0 and swa_state is not None and get_rank() == 0:
        log(f"applying SWA average over {swa_n} epochs + recalibrating BN")
        core.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
        # recalibrate BN running stats: reset and forward over source batches
        for m in core.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.reset_running_stats(); m.momentum = None  # cumulative moving average
        core.train()
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= 200:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    core(batch["feats"].to(device), batch["coords"].to(device), batch["batch_size"])
        core.eval()
        k = evaluate(core, "kitti", device, args.voxel_size, use_intensity, args.eval_max_scans or None)
        s = evaluate(core, "stf", device, args.voxel_size, use_intensity, args.eval_max_scans or None)
        log(f"  SWA EVAL: KITTI mIoU={k['mIoU']*100:.2f}  STF mIoU={s['mIoU']*100:.2f}")
        torch.save(core.state_dict(), os.path.join(args.save_path, "model.pth"))

    if get_rank() == 0:
        log("training complete")
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
