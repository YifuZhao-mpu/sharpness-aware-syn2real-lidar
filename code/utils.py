"""Label maps (verified consistent across SynLiDAR/SemanticKITTI/SemanticSTF),
mIoU meter, and small helpers. Self-contained; no external project code reused."""
import numpy as np

IGNORE = 255
NUM_CLASSES = 19
CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]

# SynLiDAR raw (0..32) -> 19-class train id (verified against PointDR synlidar.yaml labels)
SYNLIDAR_MAP = {
    0: 255, 1: 0, 2: 3, 3: 3, 4: 4, 5: 1, 6: 2, 7: 4, 8: 8, 9: 10, 10: 9,
    11: 11, 12: 5, 13: 5, 14: 5, 15: 5, 16: 6, 17: 7, 18: 12, 19: 255,
    20: 14, 21: 15, 22: 16, 23: 18, 24: 17, 25: 255, 26: 13, 27: 255,
    28: 255, 29: 255, 30: 255, 31: 255, 32: 255,
}

# SemanticKITTI raw -> 19-class train id (from official semantic-kitti.yaml learning_map)
KITTI_MAP = {
    0: 255, 1: 255, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5,
    31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: 255,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: 255,
    252: 0, 253: 6, 254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4,
}

# SemanticSTF raw -> 19-class train id (labels 1..19 = the 19 classes in order; 0/20 ignore)
STF_MAP = {i: (i - 1) for i in range(1, 20)}
STF_MAP[0] = 255
STF_MAP[20] = 255


def build_lut(mapping, max_raw=300):
    lut = np.full((max_raw,), IGNORE, dtype=np.int64)
    for k, v in mapping.items():
        lut[k] = v
    return lut


SYNLIDAR_LUT = build_lut(SYNLIDAR_MAP)
KITTI_LUT = build_lut(KITTI_MAP)
STF_LUT = build_lut(STF_MAP)


class IoUMeter:
    """Standard point-level mIoU via confusion matrix over NUM_CLASSES."""
    def __init__(self, num_classes=NUM_CLASSES, ignore=IGNORE):
        self.n = num_classes
        self.ignore = ignore
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt):
        pred = np.asarray(pred).reshape(-1)
        gt = np.asarray(gt).reshape(-1)
        valid = (gt != self.ignore) & (gt >= 0) & (gt < self.n)
        p, g = pred[valid], gt[valid]
        idx = g * self.n + p
        self.cm += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def iou_per_class(self):
        tp = np.diag(self.cm).astype(np.float64)
        fp = self.cm.sum(0) - tp
        fn = self.cm.sum(1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        return iou

    def miou(self):
        iou = self.iou_per_class()
        return float(np.nanmean(iou))

    def summary(self):
        iou = self.iou_per_class()
        return {
            "mIoU": float(np.nanmean(iou)),
            "per_class": {CLASS_NAMES[i]: (None if np.isnan(iou[i]) else round(float(iou[i]) * 100, 2)) for i in range(self.n)},
        }
