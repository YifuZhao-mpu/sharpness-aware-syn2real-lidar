"""Datasets for single-source synthetic->real LiDAR DG.
Source: SynLiDAR (train).  Targets (eval only): SemanticKITTI seq08, SemanticSTF val.
Self-contained; matches PointDR protocol (voxel 0.05, <=80k pts, MinkUNet)."""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import SYNLIDAR_LUT, KITTI_LUT, STF_LUT, IGNORE

DATA_ROOT = "/home/zyf/桌面/project/data"


def voxelize(coord, feat, label, voxel_size):
    """Quantize to voxels; return one representative point per voxel + inverse map + counts."""
    qc = np.floor(coord / voxel_size).astype(np.int64)
    qc -= qc.min(0, keepdims=True)
    key = qc[:, 0] + qc[:, 1] * 1_000_000 + qc[:, 2] * 1_000_000_000_000
    _, idx, inv = np.unique(key, return_index=True, return_inverse=True)
    counts = np.bincount(inv, minlength=idx.shape[0])
    return qc[idx], feat[idx], (label[idx] if label is not None else None), inv, idx, counts


def augment(coord):
    """Standard LiDAR train augmentation: z-rotation, scale, flip, jitter."""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    coord = coord @ R.T
    coord = coord * np.random.uniform(0.95, 1.05)
    if np.random.rand() < 0.5:
        coord[:, 0] = -coord[:, 0]
    if np.random.rand() < 0.5:
        coord[:, 1] = -coord[:, 1]
    coord = coord + np.random.normal(0, 0.02, size=coord.shape).astype(np.float32)
    return coord.astype(np.float32)


class LidarSeg(Dataset):
    def __init__(self, files, lut, voxel_size=0.05, num_points=80000,
                 train=True, use_intensity=True, label_bits=0xFFFF, intensity_scale=1.0,
                 dr_aug=False):
        self.files = files
        self.lut = lut
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.train = train
        self.use_intensity = use_intensity
        self.label_bits = label_bits
        self.intensity_scale = intensity_scale   # STF stores intensity 0-255 -> scale 1/255
        self.dr_aug = dr_aug                       # PointDR-style domain randomization baseline
        self.polarmix = False                      # PolarMix strong augmentation baseline
        self.lasermix = False                      # LaserMix strong augmentation baseline
        self.weather = False                       # LISA-style adverse-weather simulation

    def __len__(self):
        return len(self.files)

    def _load(self, bin_path):
        raw = np.fromfile(bin_path, dtype=np.float32)
        lab_path = bin_path.replace("velodyne", "labels").replace(".bin", ".label")
        rawlab = (np.fromfile(lab_path, dtype=np.int32) & self.label_bits) if os.path.exists(lab_path) else None
        # infer #cols per point from label count (SynLiDAR/KITTI=4, SemanticSTF=5)
        ncols = (raw.shape[0] // rawlab.shape[0]) if rawlab is not None and rawlab.shape[0] > 0 else 4
        scan = raw.reshape(-1, ncols)
        coord = scan[:, :3].astype(np.float32)
        inten = (scan[:, 3:4].astype(np.float32) * self.intensity_scale)
        n = coord.shape[0]
        if rawlab is not None:
            rawlab = np.clip(rawlab, 0, len(self.lut) - 1)
            label = self.lut[rawlab].astype(np.int64)
            m = min(n, label.shape[0])
            coord, inten, label = coord[:m], inten[:m], label[:m]
        else:
            label = np.full(n, 255, dtype=np.int64)
        return coord, inten, label

    def _dr(self, coord, inten, label):
        """PointDR-style domain randomization: ray-drop + intensity randomization."""
        n = coord.shape[0]
        keep = np.random.rand(n) > np.random.uniform(0.0, 0.5)
        if keep.sum() < 1000:
            keep[:] = True
        coord, inten, label = coord[keep], inten[keep], label[keep]
        inten = inten * np.random.uniform(0.5, 1.5) + \
            np.random.normal(0, 0.05, inten.shape).astype(np.float32)
        return coord, inten, label

    def _polarmix(self, coord, inten, label):
        """PolarMix-style strong augmentation (Xiao et al., NeurIPS'22): swap an azimuthal
        sector between this scan and another random source scan, plus rotate-paste a copy
        of the other scan's instance classes. A recognized strong LiDAR DG augmentation."""
        j = np.random.randint(len(self.files))
        c2, i2, l2 = self._load(self.files[j])
        if self.num_points and c2.shape[0] > self.num_points:
            s = np.random.choice(c2.shape[0], self.num_points, replace=False)
            c2, i2, l2 = c2[s], i2[s], l2[s]
        az1 = np.arctan2(coord[:, 1], coord[:, 0])
        az2 = np.arctan2(c2[:, 1], c2[:, 0])
        # (1) azimuthal sector swap: keep this scan outside a random [start,start+width] sector,
        #     take the other scan inside it
        start = np.random.uniform(-np.pi, np.pi)
        width = np.random.uniform(np.pi / 4, np.pi)
        def insec(az):
            d = np.mod(az - start, 2 * np.pi)
            return d < width
        m1 = ~insec(az1); m2 = insec(az2)
        coord = np.concatenate([coord[m1], c2[m2]], 0)
        inten = np.concatenate([inten[m1], i2[m2]], 0)
        label = np.concatenate([label[m1], l2[m2]], 0)
        # (2) instance rotate-paste: paste rotated copies of "thing" classes from the other scan
        thing = np.isin(l2, [0, 1, 2, 3, 4, 5, 6, 7])  # vehicle/person classes
        if thing.sum() > 0:
            for _ in range(2):
                th = np.random.uniform(-np.pi, np.pi)
                cs, sn = np.cos(th), np.sin(th)
                R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], dtype=np.float32)
                cp = (c2[thing] @ R.T).astype(np.float32)
                coord = np.concatenate([coord, cp], 0)
                inten = np.concatenate([inten, i2[thing]], 0)
                label = np.concatenate([label, l2[thing]], 0)
        return coord.astype(np.float32), inten.astype(np.float32), label

    def _lasermix(self, coord, inten, label):
        """LaserMix-style strong augmentation (Kong et al., CVPR'23): partition this scan
        and another random source scan into bands by INCLINATION (pitch / elevation angle)
        and alternately interleave the bands. Mixes along the laser-beam (vertical) axis,
        ORTHOGONAL to PolarMix's azimuthal mixing -> a complementary diversity route."""
        j = np.random.randint(len(self.files))
        c2, i2, l2 = self._load(self.files[j])
        if self.num_points and c2.shape[0] > self.num_points:
            s = np.random.choice(c2.shape[0], self.num_points, replace=False)
            c2, i2, l2 = c2[s], i2[s], l2[s]

        def pitch(c):
            r = np.linalg.norm(c[:, :2], axis=1) + 1e-6
            return np.arctan2(c[:, 2], r)                 # elevation angle per point
        p1, p2 = pitch(coord), pitch(c2)
        n_areas = int(np.random.choice([3, 4, 5, 6]))     # # inclination bands (LaserMix: a few areas)
        lo = float(min(p1.min(), p2.min())); hi = float(max(p1.max(), p2.max()))
        edges = np.linspace(lo, hi, n_areas + 1)[1:-1]    # interior band boundaries
        b1 = np.digitize(p1, edges); b2 = np.digitize(p2, edges)
        flip = np.random.rand() < 0.5                     # which scan owns the even bands
        take1 = (b1 % 2 == 0) if not flip else (b1 % 2 == 1)
        take2 = (b2 % 2 == 1) if not flip else (b2 % 2 == 0)
        coord = np.concatenate([coord[take1], c2[take2]], 0)
        inten = np.concatenate([inten[take1], i2[take2]], 0)
        label = np.concatenate([label[take1], l2[take2]], 0)
        return coord.astype(np.float32), inten.astype(np.float32), label

    def _weather(self, coord, inten, label):
        """LISA-inspired adverse-weather simulation (rain/fog/snow corruption of a clean
        synthetic scan) -- the weather-simulation component of UniMix-style pipelines
        (Zhao et al., CVPR'24). Randomly applies one of {rain, fog, snow}: range-dependent
        attenuation drop (farther returns lost to back-scatter), intensity reduction, and
        (fog/snow) spurious near-range scatter points labelled ignore. Bridges the
        synthetic->adverse-weather (SemanticSTF) gap on the input side."""
        n = coord.shape[0]
        rng = np.linalg.norm(coord, axis=1)
        rn = rng / (rng.max() + 1e-6)
        mode = np.random.choice(["rain", "fog", "snow"])
        alpha = {"fog": np.random.uniform(0.3, 0.7),
                 "rain": np.random.uniform(0.15, 0.4),
                 "snow": np.random.uniform(0.2, 0.5)}[mode]
        # (1) range-dependent attenuation: farther points more likely lost
        keep = np.random.rand(n) > (alpha * rn).clip(0, 0.9)
        if keep.sum() < 1000:
            keep[:] = True
        coord, inten, label = coord[keep], inten[keep], label[keep]
        # (2) intensity reduction + noise (scattering / wet surfaces lower returned power)
        inten = (inten * np.random.uniform(0.4, 0.8)
                 + np.random.normal(0, 0.03, inten.shape).astype(np.float32)).clip(0, None)
        # (3) spurious near-range scatter (fog droplets / snowflakes), labelled ignore
        if mode in ("fog", "snow"):
            ns = int(np.random.uniform(0.01, 0.05) * coord.shape[0])
            if ns > 0:
                r = np.random.uniform(1.0, 15.0, ns)
                az = np.random.uniform(-np.pi, np.pi, ns)
                el = np.random.uniform(-0.3, 0.1, ns)
                sc = np.stack([r * np.cos(el) * np.cos(az),
                               r * np.cos(el) * np.sin(az),
                               r * np.sin(el)], 1).astype(np.float32)
                coord = np.concatenate([coord, sc], 0)
                inten = np.concatenate([inten, np.random.uniform(0, 0.3, (ns, 1)).astype(np.float32)], 0)
                label = np.concatenate([label, np.full(ns, IGNORE, dtype=label.dtype)], 0)
        return coord.astype(np.float32), inten.astype(np.float32), label

    def __getitem__(self, i):
        coord, inten, label = self._load(self.files[i])
        # point subsample (train only) to bound memory
        if self.train and coord.shape[0] > self.num_points:
            sel = np.random.choice(coord.shape[0], self.num_points, replace=False)
            coord, inten, label = coord[sel], inten[sel], label[sel]
        if self.train and self.polarmix and np.random.rand() < 0.5:
            coord, inten, label = self._polarmix(coord, inten, label)
        if self.train and self.lasermix and np.random.rand() < 0.5:
            coord, inten, label = self._lasermix(coord, inten, label)
        if self.train and self.weather and np.random.rand() < 0.5:
            coord, inten, label = self._weather(coord, inten, label)
        if self.train and self.dr_aug:
            coord, inten, label = self._dr(coord, inten, label)
        rng = np.linalg.norm(coord, axis=1)                 # per-point range (pre-aug)
        if self.train:
            coord = augment(coord)
        feat_pts = np.concatenate(
            [coord, inten if self.use_intensity else np.zeros_like(inten)], 1).astype(np.float32)
        vcoord, vfeat, vlabel, inv, idx, counts = voxelize(coord, feat_pts, label, self.voxel_size)
        vnuis = np.stack([rng[idx], np.log1p(counts)], 1).astype(np.float32)  # [M,2] range, log-density
        return {
            "vcoord": torch.from_numpy(vcoord.astype(np.int32)),
            "vfeat": torch.from_numpy(vfeat),
            "vlabel": torch.from_numpy(vlabel),
            "vnuis": torch.from_numpy(vnuis),
            "plabel": torch.from_numpy(label),          # point-level labels (for eval)
            "inv": torch.from_numpy(inv.astype(np.int64)),
        }


def collate(batch):
    coords, feats, vlabels, vnuis = [], [], [], []
    plabels, invs, offsets = [], [], []
    voff = 0
    for b, s in enumerate(batch):
        n = s["vcoord"].shape[0]
        bcol = torch.full((n, 1), b, dtype=torch.int32)
        coords.append(torch.cat([bcol, s["vcoord"]], 1))
        feats.append(s["vfeat"])
        vlabels.append(s["vlabel"])
        vnuis.append(s["vnuis"])
        plabels.append(s["plabel"])
        invs.append(s["inv"] + voff)   # global voxel index for eval scatter
        offsets.append(voff)
        voff += n
    return {
        "coords": torch.cat(coords, 0),         # [Mtot,4] (batch,x,y,z)
        "feats": torch.cat(feats, 0),           # [Mtot,C]
        "vlabels": torch.cat(vlabels, 0),       # [Mtot]
        "vnuis": torch.cat(vnuis, 0),           # [Mtot,2]
        "vbatch": torch.cat([torch.full((s["vcoord"].shape[0],), b, dtype=torch.long)
                             for b, s in enumerate(batch)], 0),
        "plabels": [p for p in plabels],        # list of [Ni]
        "invs": [iv for iv in invs],            # list of [Ni] -> global voxel idx
        "offsets": offsets,
        "batch_size": len(batch),
    }


def synlidar_files(stride=None, target=19132):
    seqs = [f"{i:02d}" for i in range(13)]
    files = []
    for s in seqs:
        files += sorted(glob.glob(os.path.join(DATA_ROOT, "SynLiDAR", "sequences", s, "velodyne", "*.bin")))
    files.sort()
    if stride is None:
        stride = max(1, len(files) // target)
    return files[::stride]


def kitti_val_files():
    return sorted(glob.glob(os.path.join(
        DATA_ROOT, "semiantic_kitti", "dataset", "sequences", "08", "velodyne", "*.bin")))


def stf_val_files():
    base = os.path.join(DATA_ROOT, "SemanticSTF", "val")
    return sorted(glob.glob(os.path.join(base, "velodyne", "*.bin")))


def make_source(voxel_size=0.05, num_points=80000, stride=None, use_intensity=True,
                dr_aug=False, polarmix=False, lasermix=False, weather=False):
    ds = LidarSeg(synlidar_files(stride=stride), SYNLIDAR_LUT, voxel_size, num_points,
                  train=True, use_intensity=use_intensity, dr_aug=dr_aug)
    ds.polarmix = polarmix
    ds.lasermix = lasermix
    ds.weather = weather
    return ds


def make_target(name, voxel_size=0.05, use_intensity=True):
    if name == "kitti":
        return LidarSeg(kitti_val_files(), KITTI_LUT, voxel_size, num_points=10**9,
                        train=False, use_intensity=use_intensity)
    if name == "stf":
        return LidarSeg(stf_val_files(), STF_LUT, voxel_size, num_points=10**9,
                        train=False, use_intensity=use_intensity, intensity_scale=1.0 / 255.0)
    raise ValueError(name)
