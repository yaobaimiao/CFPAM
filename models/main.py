import torch
from torch import nn
import torch.nn.functional as F
from util import *
from models.pvtv2 import pvt_v2_b2_
from models.pvtv2 import pvt_v2_b5_
from models.Dual_ViT import dualvit_b
import matplotlib.pyplot as plt
import numpy as np
import math


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))  # , nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel / 4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel / 4), 1, kernel_size=1, stride=1, padding=0))  # , nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class AugAttentionModule(nn.Module):
    def __init__(self, input_channels=512):
        super(AugAttentionModule, self).__init__()
        self.query_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.key_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.value_transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
        )
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        x_query = self.query_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        # x_key: C,BHW
        x_key = self.key_transform(x).view(B, C, -1)  # B, C,HW
        # x_value: BHW, C
        x_value = self.value_transform(x).view(B, C, -1).permute(0, 2, 1)  # B,HW,C
        attention_bmm = torch.bmm(x_query, x_key) * self.scale  # B, HW, HW
        attention = F.softmax(attention_bmm, dim=-1)
        attention_sort = torch.sort(attention_bmm, dim=-1, descending=True)[1]
        attention_sort = torch.sort(attention_sort, dim=-1)[1]
        #####
        attention_positive_num = torch.ones_like(attention).cuda()
        attention_positive_num[attention_bmm < 0] = 0
        att_pos_mask = attention_positive_num.clone()
        attention_positive_num = torch.sum(attention_positive_num, dim=-1, keepdim=True).expand_as(attention_sort)
        attention_sort_pos = attention_sort.float().clone()
        apn = attention_positive_num - 1
        attention_sort_pos[attention_sort > apn] = 0
        attention_mask = ((attention_sort_pos + 1) ** 3) * att_pos_mask + (1 - att_pos_mask)
        out = torch.bmm(attention * attention_mask, x_value)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return out + x


class AttLayer(nn.Module):
    def __init__(self, input_channels=512):
        super(AttLayer, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (input_channels ** 0.5)
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

    def correlation(self, x5, seeds):
        B, C, H5, W5 = x5.size()
        if self.training:
            correlation_maps = F.conv2d(x5, weight=seeds)  # B,B,H,W
        else:
            correlation_maps = torch.relu(F.conv2d(x5, weight=seeds))  # B,B,H,W
        correlation_maps = correlation_maps.mean(1).view(B, -1)
        min_value = torch.min(correlation_maps, dim=1, keepdim=True)[0]
        max_value = torch.max(correlation_maps, dim=1, keepdim=True)[0]
        correlation_maps = (correlation_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[B, HW]
        correlation_maps = correlation_maps.view(B, 1, H5, W5)  # shape=[B, 1, H, W]
        return correlation_maps

    def forward(self, x5):
        # x: B,C,H,W
        x5 = self.conv(x5) + x5  # residual block
        B, C, H5, W5 = x5.size()
        x_query = self.query_transform(x5).view(B, C, -1)  # B C HW
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C)  # BHW, C
        x_key = self.key_transform(x5).view(B, C, -1)  # B,C,HW
        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1)  # C, BHW
        # W = Q^T K: B,HW,HW
        x_w1 = torch.matmul(x_query, x_key) * self.scale  # BHW, BHW
        x_w = x_w1.view(B * H5 * W5, B, H5 * W5)
        x_w = torch.max(x_w, -1).values  # BHW, B
        x_w = x_w.mean(-1)
        x_w = x_w.view(B, -1)  # B, HW
        x_w = F.softmax(x_w, dim=-1)  # B, HW
        #####  mine ######
        # x_w_max = torch.max(x_w, -1)
        # max_indices0 = x_w_max.indices.unsqueeze(-1).unsqueeze(-1)
        norm0 = F.normalize(x5, dim=1)
        # norm = norm0.view(B, C, -1)
        # max_indices = max_indices0.expand(B, C, -1)
        # seeds = torch.gather(norm, 2, max_indices).unsqueeze(-1)
        x_w = x_w.unsqueeze(1)
        x_w_max = torch.max(x_w, -1).values.unsqueeze(2).expand_as(x_w)
        mask = torch.zeros_like(x_w).cuda()
        mask[x_w == x_w_max] = 1
        mask = mask.view(B, 1, H5, W5)
        seeds = norm0 * mask
        seeds = seeds.sum(3).sum(2).unsqueeze(2).unsqueeze(3)
        cormap = self.correlation(norm0, seeds)  # M final
        x51 = x5 * cormap
        proto1 = torch.mean(x51, (0, 2, 3), True)
        return x5, proto1, x5 * proto1 + x51, x51


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.latlayer4 = LatLayer(in_channel=320)
        self.latlayer3 = LatLayer(in_channel=128)
        self.latlayer2 = LatLayer(in_channel=64)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x + y

    def forward(self, weighted_x5, x4, x3, x2, H, W):
        preds = []
        p5 = self.toplayer(weighted_x5)
        p4 = self._upsample_add(p5, self.latlayer4(x4))
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p3 = self._upsample_add(p4, self.latlayer3(x3))
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p2 = self._upsample_add(p3, self.latlayer2(x2))
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)

        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))
        return preds


# EMA-attention
class EMA_attention(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA_attention, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# ----MCAattention
class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"

        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        out = self.sigmoid(out)
        out = out.expand_as(x)

        return x * out


class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCALayer, self).__init__()

        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out


class CFPAMNet(nn.Module):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, mode='train'):
        super(CFPAMNet, self).__init__()
        self.gradients = None
        self.backbone = pvt_v2_b5_()
        self.mode = mode
        self.aug = AugAttentionModule()
        self.fusion = AttLayer(512)
        self.decoder = Decoder()
        self.MCALayer = MCALayer(512)

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, gt):
        if self.mode == 'train':
            preds = self._forward(x, gt)
        else:
            with torch.no_grad():
                preds = self._forward(x, gt)

        return preds

    def featextract(self, x):  #
        pvt = self.backbone(x)
        x2 = pvt[0]  #
        x3 = pvt[1]  #
        x4 = pvt[2]  #
        x5 = pvt[3]  #

        return x5, x4, x3, x2

    def _forward(self, x, gt):
        [B, _, H, W] = x.size()
        x5, x4, x3, x2 = self.featextract(x)  #
        feat, proto, weighted_x5, mfinal = self.fusion(x5)  #
        weighted_x5 = self.MCALayer(weighted_x5)
        feataug = self.aug(weighted_x5)
        preds = self.decoder(feataug, x4, x3, x2, H, W)
        if self.training:
            gt = F.interpolate(gt, size=weighted_x5.size()[2:], mode='bilinear', align_corners=False)
            feat_pos, proto_pos, weighted_x5_pos, mfinal_pos = self.fusion(x5 * gt)
            feat_neg, proto_neg, weighted_x5_neg, mfinal_neg = self.fusion(x5 * (1 - gt))
            return preds, proto, proto_pos, proto_neg
        return preds


class CFPAM(nn.Module):  #
    def __init__(self, mode='train'):
        super(CFPAM, self).__init__()
        set_seed(123)
        self.cfpamnet = CFPAMNet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.cfpamnet.set_mode(self.mode)

    def forward(self, x, gt):
        ########## Co-SOD ############
        preds = self.cfpamnet(x, gt)
        return preds
