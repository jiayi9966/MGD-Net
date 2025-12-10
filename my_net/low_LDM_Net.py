from .net_parts import *
from .deform_conv_v2 import DeformConv2d
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_
from .UCRNetAB_change import *
from .GBC import GBC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------- Lightweight building blocks ---------------------
class SepConv(nn.Module):
    """Depthwise separable conv: depthwise 3x3 + pointwise 1x1 (+ BN + act)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, act=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class PWConv(nn.Module):
    """Pointwise conv (1x1) + BN + act"""
    def __init__(self, in_ch, out_ch, bias=False, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


# --------------------- Modified GCM_cs_G (lightweight) ---------------------
class GCM_cs_G(nn.Module):
    def __init__(self, in_channels, out_channels, width_mult=1.0):
        """
        out_channels: int or tuple/list(4)
        width_mult: conservative scaling factor for internal "core_ch" (e.g. 0.9, 0.8)
        """
        super().__init__()

        # ---- multi-head compatibility ----
        if isinstance(out_channels, (tuple, list)):
            assert len(out_channels) == 4, "out_channels must be 4-element tuple/list"
            self.multi_out = True
            self.c1, self.c2, self.c3, self.c4 = map(int, out_channels)
            core_ch = max(self.c1, self.c2, self.c3, self.c4)
        else:
            self.multi_out = False
            core_ch = int(out_channels)

        # conservative width multiplier and lower bound
        core_ch = max(8, int(core_ch * max(0.9, width_mult)))

        pool_size = [1, 3, 5]
        # external modules assumed provided in imports: CAB, SAB, SA_Block
        self.cab = CAB(in_channels)
        self.sab = SAB()

        # multi-scale branches (lightweight)
        GClist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                PWConv(in_channels, core_ch, bias=False, act=True)
            ))
        # spatial branch keeps spatial conv but using SepConv
        GClist.append(nn.Sequential(
            SepConv(in_channels, core_ch, kernel_size=3, padding=1, bias=False),
            SA_Block(core_ch)
        ))
        self.GCmodule = nn.ModuleList(GClist)

        # reduce concatenated channels via pointwise conv
        self.GCoutmodel = nn.Sequential(
            PWConv(core_ch * 4, core_ch, bias=False, act=True)
        )

        # output heads (if multi-output)
        if self.multi_out:
            self.head1 = nn.Conv2d(core_ch, self.c1, 1, bias=False)
            self.head2 = nn.Conv2d(core_ch, self.c2, 1, bias=False)
            self.head3 = nn.Conv2d(core_ch, self.c3, 1, bias=False)
            self.head4 = nn.Conv2d(core_ch, self.c4, 1, bias=False)

    def forward(self, x):
        x = self.cab(x) * x
        x = self.sab(x) * x

        H, W = x.size(2), x.size(3)
        feats = []
        # for pooled branches: compute then upsample
        for i in range(len(self.GCmodule) - 1):
            out = self.GCmodule[i](x)
            feats.append(F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False))
        # last branch keeps spatial size
        feats.append(self.GCmodule[-1](x))

        gc = torch.cat(feats, dim=1)
        core = self.GCoutmodel(gc)

        if self.multi_out:
            y1 = self.head1(core)
            y2 = self.head2(core)
            y3 = self.head3(core)
            y4 = self.head4(core)
            return y1, y2, y3, y4
        else:
            return core


# --------------------- DecoderWithGCM_Feedback (lightweight) ---------------------
class DecoderWithGCM_Feedback(nn.Module):
    def __init__(self, in_filters2, in_filters3, in_filters4, in_filters5, lenn=1, width_mult=1.0):
        super().__init__()
        self.in_cat = in_filters2 + in_filters3 + in_filters4 + in_filters5
        # base_ch conservatively scaled from in_filters2
        base_ch = int(in_filters2 * max(0.9, width_mult))

        self.no_param_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # use SepConv (depthwise separable) in fuse layers
        self.fuse1 = nn.Sequential(
            SepConv(self.in_cat, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            PWConv(base_ch, base_ch, bias=False, act=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

        # GCM module (use width_mult to control internal size conservatively)
        self.gcm = GCM_cs_G(in_channels=in_filters2, out_channels=(in_filters2, in_filters3, in_filters4, in_filters5), width_mult=width_mult)

        self.fuse2 = nn.Sequential(
            SepConv(self.in_cat, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            PWConv(base_ch, base_ch, bias=False, act=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

    def forward(self, x2, x3, x4, x5):
        # upsample smaller scales to x2's spatial resolution
        x3_up = self.no_param_up(x3)
        x4_up = self.no_param_up(self.no_param_up(x4))
        x5_up = self.no_param_up(self.no_param_up(self.no_param_up(x5)))

        # first fusion
        fused_h8 = self.fuse1(torch.cat([x2, x4_up, x3_up, x5_up], dim=1))
        fused_h8 = self.mlp1(fused_h8)

        # GCM enhancement
        g2, g3, g4, g5 = self.gcm(fused_h8)

        # residual feedback
        x2_ = x2 + g2
        x3_ = x3_up + g3
        x4_ = x4_up + g4
        x5_ = x5_up + g5

        # second fusion
        fused_h8_ref = self.fuse2(torch.cat([x2_, x3_, x4_, x5_], dim=1))
        fused_h8_ref = self.mlp2(fused_h8_ref)

        return fused_h8_ref


# --------------------- Decoder2 (use upsample + 1x1 instead of ConvTranspose) ---------------------
class Decoder2(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, in_ch3=0, in_ch4=0, attn=False, width_mult=1.0):
        super().__init__()
        # replace transpose conv by upsample + pointwise proj
        self.deconv_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.deconv_proj = PWConv(in_ch4, out_ch, bias=False, act=True)

        # feedback fusion branch
        self.out = DecoderWithGCM_Feedback(in_ch4,in_ch3,in_ch2, in_ch1, width_mult=width_mult)
        self.no_param_up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # concatenation channels = x5 + d2
        c = in_ch4 + out_ch

        # keep your DoubleConv / Residual / SE modules
        self.conv = DoubleConv(c, out_ch)
        self.residual = Residual(c, c)
        self.se = SEWeightModule(c)

    def forward(self, x1, x5, x2, x3, x4):
        # feedback branch produces H/8 feature
        d2 = self.out(x4, x3, x2, x1)

        # upsample + project
        d2 = self.deconv_up(d2)
        d2 = self.deconv_proj(d2)

        # fuse with main branch x5
        c = torch.cat([x5, d2], dim=1)
        w = self.se(c) * c
        c = self.residual(w)
        fuse = self.conv(c)

        return fuse


# --------------------- MDUNET variants (with width_mult) ---------------------
class MDUNET(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2, n_filts=64, width_mult=1.0):
        super(MDUNET, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.GBC_1 = GBC(filters[0])
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.GBC_2 = GBC(filters[0])
        self.encoder2 = resnet.layer2
        self.GBC_3 = GBC(filters[1])
        self.encoder3 = resnet.layer3
        self.GBC_4 = GBC(filters[2])
        self.encoder4 = resnet.layer4
        self.GBC_5 = GBC(filters[3])

        # decoder (pass width_mult)
        self.decoder = Decoder2(in_ch1=512, out_ch=64, in_ch2=256, in_ch3=128, in_ch4=64, width_mult=width_mult)

        # final head: upsample + 1x1 proj
        self.finaldeconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.finalproj = PWConv(filters[0], 32, bias=False, act=True)
        self.GBC_o = GBC(32)
        self.finalrelu1 = nonlinearity
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        try:
            self.GBC_1(x_0)
        except Exception:
            pass
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        try:
            e1 = self.GBC_2(e1)
        except Exception:
            pass
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)
        try:
            e2 = self.GBC_3(e2)
        except Exception:
            pass
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)
        try:
            e3 = self.GBC_4(e3)
        except Exception:
            pass
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)
        try:
            e4 = self.GBC_5(e4)
        except Exception:
            pass
        e4 = self.drop(e4)

        up4 = self.decoder(e4, x_0, e3, e2, e1)

        out = self.finaldeconv1(up4)
        out = self.finalproj(out)
        out = self.finalrelu1(out)
        try:
            out = self.GBC_o(out)
        except Exception:
            pass
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class MDUNET_l1(MDUNET):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2, n_filts=64, width_mult=1.0):
        super().__init__(num_classes=num_classes, BatchNorm=BatchNorm, drop_rate=drop_rate, n_filts=n_filts, width_mult=width_mult)


class MDUNET_de(MDUNET):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2, n_filts=64, width_mult=1.0):
        super(MDUNET_de, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1

        self.encoder2 = resnet.layer2

        self.encoder3 = resnet.layer3

        self.encoder4 = resnet.layer4


        # decoder (pass width_mult)
        self.decoder = Decoder2(in_ch1=512, out_ch=64, in_ch2=256, in_ch3=128, in_ch4=64, width_mult=width_mult)

        # final head: upsample + 1x1 proj
        self.finaldeconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.finalproj = PWConv(filters[0], 32, bias=False, act=True)

        self.finalrelu1 = nonlinearity
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)

        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)

        e1 = self.drop(e1)
        e2 = self.encoder2(e1)

        e2 = self.drop(e2)
        e3 = self.encoder3(e2)

        e3 = self.drop(e3)
        e4 = self.encoder4(e3)

        e4 = self.drop(e4)

        up4 = self.decoder(e4, x_0, e3, e2, e1)

        out = self.finaldeconv1(up4)
        out = self.finalproj(out)
        out = self.finalrelu1(out)

        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


# --------------------- Utility: flops & params check (optional) ---------------------
# Uncomment and use thop to profile:
# from thop import profile
# from thop import clever_format
# if __name__ == '__main__':
#     m = MDUNET_de(num_classes=1, width_mult=0.9).cuda()
#     input = torch.randn(1,3,576,576).cuda()
#     macs, params = profile(m, inputs=(input, ))
#     macs, params = clever_format([macs, params], '%.3f')
#     print('MACs:', macs, 'Params:', params)
