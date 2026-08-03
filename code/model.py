"""MinkUNet (spconv) for LiDAR semantic segmentation, channel-ratio (cr) scalable.
Self-contained re-implementation of the standard SparseConv UNet. Captures
early-stage features for the sensing-style whitening regularizer."""
from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import spconv.pytorch as spconv


def norm_fn(c):
    return nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)


class BasicBlock(spconv.SparseModule):
    def __init__(self, in_c, out_c, indice_key=None):
        super().__init__()
        if in_c == out_c:
            self.proj = nn.Identity()
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_c, out_c, 1, bias=False), norm_fn(out_c))
        self.conv1 = spconv.SubMConv3d(in_c, out_c, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(out_c)
        self.conv2 = spconv.SubMConv3d(out_c, out_c, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.proj(x) if not isinstance(self.proj, nn.Identity) else x
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity.features))
        return out


class MinkUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=19, cr=0.5,
                 base_channels=32, channels=(32, 64, 128, 256, 256, 128, 96, 96),
                 layers=(2, 3, 4, 6, 2, 2, 2, 2), capture_stages=(0, 1)):
        super().__init__()
        base_channels = int(base_channels * cr)
        channels = tuple(int(c * cr) for c in channels)
        self.num_stages = len(layers) // 2
        self.num_classes = num_classes
        self.capture_stages = set(capture_stages)
        self.capture = {}

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, 3, padding=1, bias=False, indice_key="stem"),
            norm_fn(base_channels), nn.ReLU())

        enc_c = base_channels
        dec_c = channels[-1]
        self.down, self.up, self.enc, self.dec = (nn.ModuleList() for _ in range(4))
        for s in range(self.num_stages):
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(enc_c, channels[s], 2, stride=2, bias=False, indice_key=f"spconv{s+1}"),
                norm_fn(channels[s]), nn.ReLU()))
            self.enc.append(spconv.SparseSequential(OrderedDict(
                [(f"block{i}", BasicBlock(channels[s], channels[s], indice_key=f"subm{s+1}"))
                 for i in range(layers[s])])))
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(channels[len(channels)-s-2], dec_c, 2, bias=False, indice_key=f"spconv{s+1}"),
                norm_fn(dec_c), nn.ReLU()))
            blocks = []
            for i in range(layers[len(channels)-s-1]):
                ic = dec_c + enc_c if i == 0 else dec_c
                blocks.append((f"block{i}", BasicBlock(ic, dec_c, indice_key=f"subm{s}")))
            self.dec.append(spconv.SparseSequential(OrderedDict(blocks)))
            enc_c = channels[s]
            dec_c = channels[len(channels)-s-2]

        self.final = spconv.SubMConv3d(channels[-1], num_classes, 1, bias=True)
        self.enc_channels_list = [base_channels] + [channels[s] for s in range(self.num_stages)]
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, spconv.SubMConv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feats, coords, batch_size, return_feat=False):
        # coords: [M,4] int (batch,x,y,z)
        sparse_shape = (coords[:, 1:].max(0).values + 96).tolist()
        x = spconv.SparseConvTensor(feats, coords.int().contiguous(), sparse_shape, batch_size)
        x = self.conv_input(x)
        # stem features align 1-1 with input voxel order (SubMConv preserves order)
        self.capture = {"stem": x.features}
        skips = [x]
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            if s in self.capture_stages:
                self.capture[s] = (x.features, x.indices[:, 0].long())  # (feat[M,C], batch_idx[M])
            skips.append(x)
        x = skips.pop(-1)
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)
        feat = x.features                 # decoder features (pre-final), for PointDR prototypes
        x = self.final(x)
        if return_feat:
            return x.features, feat
        return x.features  # [Mtot, num_classes] voxel logits aligned with input voxel order
