import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SubNet(nn.Module):

    def __init__(self, num_channels=3, num_feature=16, num_blocks=5):
        super(SubNet, self).__init__()
        self.num_blocks = num_blocks

        # ---- 提取特征层 ----
        self.conv0 = nn.Conv2d(num_channels, num_feature, kernel_size=3, padding=1)

        # ---- 递归块 ----
        self.conv1 = nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_feature, num_feature, kernel_size=1)
        self.conv3 = nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1)

        # ---- 重建层  ----
        self.conv4 = nn.Conv2d(num_feature, num_channels, kernel_size=1)

    def forward(self, x):

        out0 = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        out_block = out0

        for _ in range(self.num_blocks):
            out = F.leaky_relu(self.conv1(out_block), negative_slope=0.2)
            out = F.leaky_relu(self.conv2(out),       negative_slope=0.2)
            out = F.leaky_relu(self.conv3(out),       negative_slope=0.2)
            out_block = out + out0

        return self.conv4(out_block) + x


class LPNet(nn.Module):

    def __init__(self, num_pyramids=5, num_blocks=5, num_feature=16, num_channels=3):
        super(LPNet, self).__init__()
        self.num_pyramids = num_pyramids
        self.num_channels = num_channels

        # ---------- 固定高斯核 ----------
        k = np.float32([.0625, .25, .375, .25, .0625])  # type: ignore
        k = np.outer(k, k)
        gk = (k / k.sum()).astype(np.float32)
        # shape: [C, 1, 5, 5]  用于 groups=C 的深度卷积
        gk_tensor = torch.from_numpy(gk).unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)
        self.register_buffer('gk', gk_tensor)

        # ---------- 子网络 ----------
        feat = [max(1, num_feature // (2 ** (4 - i))) if i < 4 else num_feature
                for i in range(num_pyramids)]
        # 即 [1, 2, 4, 8, 16]
        self.subnets = nn.ModuleList(
            [SubNet(num_channels, f, num_blocks) for f in feat]
        )

    @staticmethod
    def _pad_same(x, ksize, stride):
        h, w = x.shape[2], x.shape[3]
        oh = (h + stride - 1) // stride
        ow = (w + stride - 1) // stride
        ph = max((oh - 1) * stride + ksize - h, 0)
        pw = max((ow - 1) * stride + ksize - w, 0)
        if ph > 0 or pw > 0:
            x = F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2])
        return x

    def _down(self, img):
        img = self._pad_same(img, 5, 2)
        return F.conv2d(img, self.gk, stride=2, groups=self.num_channels)   # type: ignore

    def _up(self, img, target_shape):
        th, tw = target_shape[2], target_shape[3]
        oh = max(0, min(1, th - (img.shape[2] * 2 - 1)))
        ow = max(0, min(1, tw - (img.shape[3] * 2 - 1)))
        return F.conv_transpose2d(img, self.gk * 4, stride=2, padding=2, # type: ignore
                                  output_padding=(oh, ow), groups=self.num_channels)

    def _laplacian_pyramid(self, img, n):
        levels = []
        cur = img
        for _ in range(n):
            low = self._down(cur)
            low_up = self._up(low, cur.shape)
            levels.append(cur - low_up)
            cur = low
        levels.append(cur)
        return levels[::-1]#返回的是从最低分辨率到最高分辨率的纹理图

    def forward(self, x):
        pyramid = self._laplacian_pyramid(x, self.num_pyramids - 1)
        outputs, prev_up = [], None
        for i in range(self.num_pyramids):
            out = self.subnets[i](pyramid[i])
            if prev_up is not None:
                out = out + prev_up
            out = F.relu(out)
            outputs.append(out)
            if i < self.num_pyramids - 1:
                prev_up = self._up(out, pyramid[i + 1].shape)
        return outputs


class LPNet_CrackSegmenter(nn.Module):

    def __init__(self, pretrained_lpnet):
        super(LPNet_CrackSegmenter, self).__init__()
        self.backbone = pretrained_lpnet
        self.seg_head = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

    def forward(self, x):
        outputs = self.backbone(x)
        x_reco = outputs[-1]
        mask_logits = self.seg_head(x_reco)
        return mask_logits
