from .vit_seg_modeling_resnet_skip import *
from .net_parts import *
from .deform_conv_v2 import DeformConv2d
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_
from .UCRNetAB_change import *
from .GBC import GBC

class GCM_cs_G(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        out_channels:
            - int:    单输出，返回 [B, out_channels, H, W]
            - tuple/list(4): 多输出，返回 (y1, y2, y3, y4)，
              其中 yk.shape[1] 分别等于 out_channels[k-1]
        """
        super().__init__()

        # ---- 兼容：判断是否多头 ----
        if isinstance(out_channels, (tuple, list)):
            assert len(out_channels) == 4, "out_channels 需为4个通道大小 (c1, c2, c3, c4)"
            self.multi_out = True
            self.c1, self.c2, self.c3, self.c4 = map(int, out_channels)
            core_ch = max(self.c1, self.c2, self.c3, self.c4)  # 核心通道取最大，避免瓶颈
        else:
            self.multi_out = False
            core_ch = int(out_channels)

        pool_size = [1, 3, 5]
        self.cab = CAB(in_channels)
        self.sab = SAB()

        # 多尺度分支：每个分支输出 core_ch
        GClist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, core_ch, 1, 1, bias=False),
                nn.GELU()
            ))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, core_ch, 1, 1, bias=False),
            nn.GELU(),
            SA_Block(core_ch)
        ))
        self.GCmodule = nn.ModuleList(GClist)

        # 4 分支 concat 后用 1x1 压回 core_ch
        self.GCoutmodel = nn.Sequential(
            nn.Conv2d(core_ch * 4, core_ch, kernel_size=1, bias=False),
            nn.GELU()
        )

        # 多头 1x1 输出头
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
        for i in range(len(self.GCmodule) - 1):
            feats.append(F.interpolate(self.GCmodule[i](x), size=(H, W),
                                       mode='bilinear', align_corners=True))
        feats.append(self.GCmodule[-1](x))
        gc = torch.cat(feats, dim=1)   # [B, core_ch*4, H, W]
        core = self.GCoutmodel(gc)     # [B, core_ch,   H, W]

        if self.multi_out:
            y1 = self.head1(core)      # [B, c1, H, W]
            y2 = self.head2(core)      # [B, c2, H, W]
            y3 = self.head3(core)      # [B, c3, H, W]
            y4 = self.head4(core)      # [B, c4, H, W]
            return y1, y2, y3, y4
        else:
            return core


class GG(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        out_channels:
            - int:    单输出，返回 [B, out_channels, H, W]
            - tuple/list(4): 多输出，返回 (y1, y2, y3, y4)，
              其中 yk.shape[1] 分别等于 out_channels[k-1]
        """
        super().__init__()

        # ---- 判断是否多头 ----
        if isinstance(out_channels, (tuple, list)):
            assert len(out_channels) == 4, "out_channels 需为4个通道大小 (c1, c2, c3, c4)"
            self.multi_out = True
            self.c1, self.c2, self.c3, self.c4 = map(int, out_channels)
            core_ch = max(self.c1, self.c2, self.c3, self.c4)
        else:
            self.multi_out = False
            core_ch = int(out_channels)

        # 核心卷积（简化版本，不用 CAB、SAB）
        self.conv1 = nn.Conv2d(in_channels, core_ch, 1, bias=False)
        self.act = nn.GELU()

        # 多头输出
        if self.multi_out:
            self.head1 = nn.Conv2d(core_ch, self.c1, 1, bias=False)
            self.head2 = nn.Conv2d(core_ch, self.c2, 1, bias=False)
            self.head3 = nn.Conv2d(core_ch, self.c3, 1, bias=False)
            self.head4 = nn.Conv2d(core_ch, self.c4, 1, bias=False)

    def forward(self, x, sizes=None):
        """
        x: 输入特征 [B, C, H, W]
        sizes: tuple/list，目标输出的尺寸，如 [(H1, W1), (H2, W2), (H3, W3), (H4, W4)]
        """
        core = self.act(self.conv1(x))  # [B, core_ch, H, W]

        if self.multi_out:
            y1 = self.head1(core)
            y2 = self.head2(core)
            y3 = self.head3(core)
            y4 = self.head4(core)

            # 自动对齐尺寸
            if sizes is not None:
                y1 = F.interpolate(y1, size=sizes[0], mode="bilinear", align_corners=False)
                y2 = F.interpolate(y2, size=sizes[1], mode="bilinear", align_corners=False)
                y3 = F.interpolate(y3, size=sizes[2], mode="bilinear", align_corners=False)
                y4 = F.interpolate(y4, size=sizes[3], mode="bilinear", align_corners=False)

            return y1, y2, y3, y4
        else:
            return core

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCM_cs_G_(nn.Module):
    """
    修改版：合并原先多分支输出为单输出，且保持输出空间尺寸与输入一致。
    用法与原类类似，但最终返回单个张量（[B, out_ch, H, W]）。
    """
    def __init__(self, in_channels, out_channels):
        super(GCM_cs_G_, self).__init__()

        # 如果传入的是 tuple/list（原先支持多输出），这里容错处理：
        if isinstance(out_channels, (tuple, list)):
            assert len(out_channels) == 4, "如果传入 list/tuple，长度应为4"
            # final_out_ch 使用第一个元素（可按需修改为其它策略）
            self.final_out_ch = int(out_channels[0])
            core_ch = max(map(int, out_channels))
        else:
            self.final_out_ch = int(out_channels)
            core_ch = int(out_channels)

        pool_size = [1, 3, 5]
        # 保留 CAB/SAB（如果你已经在其他地方定义了 CAB/SAB/SA_Block）
        self.cab = CAB(in_channels)
        self.sab = SAB()

        # 多尺度分支：每个分支输出 core_ch（先做 pool / 1x1）
        GClist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, core_ch, kernel_size=1, bias=False),
                nn.GELU()
            ))
        # 最后一个分支保持原始空间尺寸并使用 SA_Block（如有）
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, core_ch, kernel_size=1, bias=False),
            nn.GELU(),
            SA_Block(core_ch)  # 若没有 SA_Block，请替换或移除
        ))
        self.GCmodule = nn.ModuleList(GClist)

        # 将 concat 后的 core_ch*4 投影为最终单一输出通道 self.final_out_ch
        self.out_proj = nn.Sequential(
            nn.Conv2d(core_ch * 4, core_ch, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(core_ch, self.final_out_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # x: [B, C_in, H, W]
        # keep CAB/SAB behavior as original
        x = self.cab(x) * x
        x = self.sab(x) * x

        H, W = x.size(2), x.size(3)
        feats = []
        # 对池化分支做回插值对齐到输入尺寸（与原实现一致）
        for i in range(len(self.GCmodule) - 1):
            out = self.GCmodule[i](x)
            feats.append(F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False))
        # 最后一分支保持原空间
        feats.append(self.GCmodule[-1](x))

        gc = torch.cat(feats, dim=1)   # [B, core_ch*4, H, W]
        out = self.out_proj(gc)        # [B, final_out_ch, H, W]

        return out

# ----------------------------------------
# DecoderWithGCM_Feedback（三路：F2/F3/F4）
class DecoderWithGCM_Feedback(nn.Module):
    def __init__(self, in_filters2, in_filters3, in_filters4,in_filters5, lenn=1):
        super().__init__()
        self.in_cat = in_filters2 + in_filters3 + in_filters4+in_filters5
        base_ch = in_filters2  # ← H/8 的通道作为基准

        self.no_param_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 第一次 H/8 融合 → base_ch
        self.fuse1 = nn.Sequential(
            nn.Conv2d(self.in_cat, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

        # GCM：in/out 都是 base_ch（= H/8 通道）
        self.gcm = GG(in_channels=in_filters2, out_channels=(in_filters2, in_filters3, in_filters4,in_filters5))

        # 第二次 H/8 融合 → base_ch
        self.fuse2 = nn.Sequential(
            nn.Conv2d(self.in_cat, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

        # （可选）兼容老写法，防止有人还写 self.GCM_cs_G(...)
        # self.GCM_cs_G = self.gcm

    def forward(self, x2, x3, x4,x5):
        """
        x2: [B, C2, H/8,  W/8]
        x3: [B, C3, H/16, W/16]
        x4: [B, C4, H/32, W/32]
        """
        # 统一到 H/4
        x3_up = self.no_param_up(x3)
        x4_up = self.no_param_up(self.no_param_up(x4))               # H/16 -> H/8
        x5_up = self.no_param_up(self.no_param_up(self.no_param_up(x5)))  # H/32 -> H/16 -> H/8

        # 第一次融合 (H/4)
        fused_h8 = self.fuse1(torch.cat([x2,x4_up, x3_up, x5_up], dim=1))
        fused_h8 = self.mlp1(fused_h8)

        # GCM 增强（H/8 内部）
        g2, g3, g4,g5 = self.gcm(fused_h8)

        # 反馈（残差）
        x2_ = x2 + g2
        x3_ = x3_up + g3
        x4_ = x4_up + g4
        x5_ = x5_up +g5

        # 第二次融合 (H/8)
        fused_h8_ref = self.fuse2(torch.cat([x2_, x3_, x4_,x5_], dim=1))
        fused_h8_ref = self.mlp2(fused_h8_ref)     # [B, C2, H/8, W/8]

        return fused_h8_ref


class Decoder_Feedback_G(nn.Module):
    def __init__(self, in_filters2, in_filters3, in_filters4,in_filters5, lenn=1):
        super().__init__()
        self.in_cat = in_filters2 + in_filters3 + in_filters4+in_filters5
        base_ch = in_filters2  # ← H/8 的通道作为基准

        self.no_param_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 第一次 H/8 融合 → base_ch
        self.fuse1 = nn.Sequential(
            nn.Conv2d(self.in_cat, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

        # GCM：in/out 都是 base_ch（= H/8 通道）
        self.gcm = GCM_cs_G_(in_filters2,in_filters2)

        # 第二次 H/8 融合 → base_ch
        self.fuse2 = nn.Sequential(
            nn.Conv2d(self.in_cat, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

        # （可选）兼容老写法，防止有人还写 self.GCM_cs_G(...)
        # self.GCM_cs_G = self.gcm

    def forward(self, x2, x3, x4,x5):
        """
        x2: [B, C2, H/8,  W/8]
        x3: [B, C3, H/16, W/16]
        x4: [B, C4, H/32, W/32]
        """
        # 统一到 H/4
        x3_up = self.no_param_up(x3)
        x4_up = self.no_param_up(self.no_param_up(x4))               # H/16 -> H/8
        x5_up = self.no_param_up(self.no_param_up(self.no_param_up(x5)))  # H/32 -> H/16 -> H/8

        # # 第一次融合 (H/4)
        # fused_h8 = self.fuse1(torch.cat([x2,x4_up, x3_up, x5_up], dim=1))
        # fused_h8 = self.mlp1(fused_h8)
        #
        # # GCM 增强（H/8 内部）
        # # g2, g3, g4,g5 = self.gcm(fused_h8)
        #
        # # 反馈（残差）
        # x2_ = x2 + g2
        # x3_ = x3_up + g3
        # x4_ = x4_up + g4
        # x5_ = x5_up +g5

        # 第二次融合 (H/8)
        fused_h8_ref = self.fuse2(torch.cat([x2, x3_up, x4_up,x5_up], dim=1))
        fused_h8_ref = self.mlp2(fused_h8_ref)     # [B, C2, H/8, W/8]
        out=self.gcm(fused_h8_ref)

        return out
class Decoder_Feedback(nn.Module):
    def __init__(self, in_filters2, in_filters3, in_filters4,in_filters5, lenn=1):
        super().__init__()
        self.in_cat = in_filters2 + in_filters3 + in_filters4+in_filters5
        base_ch = in_filters2  # ← H/8 的通道作为基准

        self.no_param_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 第一次 H/8 融合 → base_ch
        self.fuse1 = nn.Sequential(
            nn.Conv2d(self.in_cat, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

        # GCM：in/out 都是 base_ch（= H/8 通道）
        # self.gcm = GG(in_channels=in_filters2, out_channels=(in_filters2, in_filters3, in_filters4,in_filters5))

        # 第二次 H/8 融合 → base_ch
        self.fuse2 = nn.Sequential(
            nn.Conv2d(self.in_cat, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1, bias=True),
        )

        # （可选）兼容老写法，防止有人还写 self.GCM_cs_G(...)
        # self.GCM_cs_G = self.gcm

    def forward(self, x2, x3, x4,x5):
        """
        x2: [B, C2, H/8,  W/8]
        x3: [B, C3, H/16, W/16]
        x4: [B, C4, H/32, W/32]
        """
        # 统一到 H/4
        x3_up = self.no_param_up(x3)
        x4_up = self.no_param_up(self.no_param_up(x4))               # H/16 -> H/8
        x5_up = self.no_param_up(self.no_param_up(self.no_param_up(x5)))  # H/32 -> H/16 -> H/8

        # # 第一次融合 (H/4)
        # fused_h8 = self.fuse1(torch.cat([x2,x4_up, x3_up, x5_up], dim=1))
        # fused_h8 = self.mlp1(fused_h8)
        #
        # # GCM 增强（H/8 内部）
        # # g2, g3, g4,g5 = self.gcm(fused_h8)
        #
        # # 反馈（残差）
        # x2_ = x2 + g2
        # x3_ = x3_up + g3
        # x4_ = x4_up + g4
        # x5_ = x5_up +g5

        # 第二次融合 (H/8)
        fused_h8_ref = self.fuse2(torch.cat([x2, x3_up, x4_up,x5_up], dim=1))
        fused_h8_ref = self.mlp2(fused_h8_ref)     # [B, C2, H/8, W/8]

        return fused_h8_ref
# ----------------------------
# Decoder2（单级解码，保持你的结构）
# ----------------------------
class Decoder2(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, in_ch3=0, in_ch4=0, attn=False):
        super().__init__()
        # 上采主分支：e4 -> H/16（如果 e4 是 H/32），或 e4 -> H/8（若前面已有一次池化）
        self.deconv = nn.ConvTranspose2d(in_ch4, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nonlinearity

        # 反馈分支（F2/F3/F4 融合于 H/8）
        # self.out = DecoderWithGCM_Feedback(in_ch4,in_ch3,in_ch2, in_ch1)  #Decoder2(in_ch1=512, out_ch=64, in_ch2=256, in_ch3=128, in_ch4=64)
        self.out =Decoder_Feedback_G(in_ch4,in_ch3,in_ch2, in_ch1)
        self.no_param_up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 拼接通道 = 上采主分支(out_ch) + 反馈分支(in_ch2)
        c = in_ch4 + out_ch

        self.conv = DoubleConv(c, out_ch)
        self.residual = Residual(c, c)
        self.se = SEWeightModule(c)

    def forward(self, x1, x5,x2, x3, x4):
        # 主分支：e4 -> 上采

        # 反馈分支：F2/F3/F4 → H/8 融合
        d2 = self.out(x4,x3, x2, x1)   # [B, in_ch2, H/8, W/8]x1, x2, x3, x4

        d2 = self.deconv(d2)
        d2 = self.norm(d2)
        d2 = self.relu(d2)
        # d2 = self.no_param_up(d2)   # 与主分支 x1 对齐（若 x1 是 H/4 则看你上游尺度，这里保持你原写法）
        # print("d2:",d2)
        # 融合 + 注意力/残差 + 双卷积
        c = torch.cat([x5, d2], dim=1)
        w = self.se(c) * c
        c = self.residual(w)
        fuse = self.conv(c)

        return fuse


# ----------------------------
# MDUNET 主体（保持你的调用方式）
# ----------------------------
class MDUNET(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2, n_filts=64):
        super(MDUNET, self).__init__()

        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.GBC_1 = GBC(filters[0])
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1   # 输出 C=64,   H/4
        self.GBC_2 = GBC(filters[0])
        self.encoder2 = resnet.layer2   # 输出 C=128,  H/8
        self.GBC_3 = GBC(filters[1])
        self.encoder3 = resnet.layer3   # 输出 C=256,  H/16
        self.GBC_4 = GBC(filters[2])
        self.encoder4 = resnet.layer4   # 输出 C=512,  H/32
        self.GBC_5 = GBC(filters[3])

        # 单级解码器（保持你的通道参数）
        self.decoder = Decoder2(in_ch1=512, out_ch=64, in_ch2=256, in_ch3=128, in_ch4=64)

        # 最后上采 + 头
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 64 -> 32, 上采一倍
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.GBC_o = GBC(32)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        self.GBC_1(x_0)
        x = self.firstmaxpool(x)    # H -> H/4

        e1 = self.encoder1(x)         # C=64,  H/4
        e1 = self.GBC_2(e1)
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)        # C=128, H/8
        e2 = self.GBC_3(e2)
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)        # C=256, H/16
        e3 = self.GBC_4(e3)
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)        # C=512, H/32
        e4 = self.GBC_5(e4)
        e4 = self.drop(e4)

        # Decoder（保持你的单级结构与输入顺序）
        up4 = self.decoder(e4,x_0, e3, e2, e1)  # 输出 C=64，空间尺度与你的 deconv/upsample 对齐x1, x2, x3, x4

        # Head
        out = self.finaldeconv1(up4)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        out=self.GBC_o(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class MDUNET_l1(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2, n_filts=64):
        super(MDUNET_l1, self).__init__()

        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.GBC_1 = GBC(filters[0])
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1   # 输出 C=64,   H/4

        self.encoder2 = resnet.layer2   # 输出 C=128,  H/8

        self.encoder3 = resnet.layer3   # 输出 C=256,  H/16

        self.encoder4 = resnet.layer4   # 输出 C=512,  H/32


        # 单级解码器（保持你的通道参数）
        self.decoder = Decoder2(in_ch1=512, out_ch=64, in_ch2=256, in_ch3=128, in_ch4=64)

        # 最后上采 + 头
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 64 -> 32, 上采一倍
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.GBC_o = GBC(32)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        self.GBC_1(x_0)
        x = self.firstmaxpool(x)    # H -> H/4
        e1 = self.encoder1(x)         # C=64,  H/4
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)        # C=128, H/8
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)        # C=256, H/16
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)        # C=512, H/32
        e4 = self.drop(e4)

        # Decoder（保持你的单级结构与输入顺序）
        up4 = self.decoder(e4,x_0, e3, e2, e1)  # 输出 C=64，空间尺度与你的 deconv/upsample 对齐x1, x2, x3, x4

        # Head
        out = self.finaldeconv1(up4)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        out=self.GBC_o(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class MDUNET_de(nn.Module):
    def __init__(self, num_classes=1, BatchNorm=nn.BatchNorm2d, drop_rate=0.2, n_filts=64):
        super(MDUNET_de, self).__init__()

        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        # self.GBC_1 = GBC(filters[0])
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1   # 输出 C=64,   H/4

        self.encoder2 = resnet.layer2   # 输出 C=128,  H/8

        self.encoder3 = resnet.layer3   # 输出 C=256,  H/16

        self.encoder4 = resnet.layer4   # 输出 C=512,  H/32


        # 单级解码器（保持你的通道参数）
        self.decoder = Decoder2(in_ch1=512, out_ch=64, in_ch2=256, in_ch3=128, in_ch4=64)

        # 最后上采 + 头
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 64 -> 32, 上采一倍
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.GBC_o = GBC(32)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x_0 = self.firstrelu(x)
        # self.GBC_1(x_0)
        x = self.firstmaxpool(x)    # H -> H/4
        e1 = self.encoder1(x)         # C=64,  H/4
        e1 = self.drop(e1)
        e2 = self.encoder2(e1)        # C=128, H/8
        e2 = self.drop(e2)
        e3 = self.encoder3(e2)        # C=256, H/16
        e3 = self.drop(e3)
        e4 = self.encoder4(e3)        # C=512, H/32
        e4 = self.drop(e4)

        # Decoder（保持你的单级结构与输入顺序）
        up4 = self.decoder(e4,x_0, e3, e2, e1)  # 输出 C=64，空间尺度与你的 deconv/upsample 对齐x1, x2, x3, x4

        # Head
        out = self.finaldeconv1(up4)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        # out=self.GBC_o(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out



