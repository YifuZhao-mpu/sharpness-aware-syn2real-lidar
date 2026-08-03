"""Sensing-Style Whitening (SSW): decorrelate early sparse-conv feature channels,
selectively emphasizing channels coupled to LiDAR sensing nuisance (range, local
density). Training-time only; identity at inference. Modes:
  none    -> no regularization (source-only)
  blunt   -> whiten ALL channel correlations (RobustNet-style decorrelation)
  random  -> whiten a random channel subset (selectivity control)
  sensing -> whiten channels coupled to sensing nuisance (ours)
"""
import torch

EPS = 1e-5


def _standardize(F):
    Fc = F - F.mean(0, keepdim=True)
    return Fc / (Fc.std(0, keepdim=True) + EPS)


def whitening_loss(feats, vbatch, vnuis, mode="sensing", random_frac=0.5):
    if mode == "none":
        return feats.new_zeros(())
    C = feats.shape[1]
    B = int(vbatch.max().item()) + 1
    losses = []
    for b in range(B):
        m = vbatch == b
        F = feats[m]
        if F.shape[0] < C + 2:
            continue
        Fh = _standardize(F)                              # [Mb, C]
        cov = (Fh.t() @ Fh) / (Fh.shape[0] - 1)           # [C, C]
        off = cov - torch.diag(torch.diag(cov))
        if mode == "blunt":
            w = torch.ones(C, device=feats.device)
        elif mode == "random":
            w = (torch.rand(C, device=feats.device) < random_frac).float() + EPS
        elif mode == "sensing":
            zc = _standardize(vnuis[m])                   # [Mb, K]
            corr = (Fh.t() @ zc) / (Fh.shape[0] - 1)      # [C, K]
            w = corr.abs().max(1).values.detach()         # nuisance-coupling per channel
            w = w / (w.max() + EPS)
        else:
            raise ValueError(mode)
        W = torch.outer(w, w)
        denom = (w.sum() ** 2 - (w ** 2).sum()).clamp_min(EPS)  # off-diag weight mass
        losses.append((W * off.pow(2)).sum() / denom)
    if not losses:
        return feats.new_zeros(())
    return torch.stack(losses).mean()


@torch.no_grad()
def nuisance_corr(feats, vbatch, vnuis):
    """Diagnostic: mean over channels of max-abs linear correlation with sensing nuisance.
    High value => features linearly encode sensing nuisance (domain-variant)."""
    C = feats.shape[1]
    B = int(vbatch.max().item()) + 1
    vals = []
    for b in range(B):
        m = vbatch == b
        F = feats[m]
        if F.shape[0] < C + 2:
            continue
        Fh = _standardize(F)
        zc = _standardize(vnuis[m])
        corr = (Fh.t() @ zc) / (Fh.shape[0] - 1)
        vals.append(corr.abs().max(1).values.mean())
    return torch.stack(vals).mean().item() if vals else 0.0
