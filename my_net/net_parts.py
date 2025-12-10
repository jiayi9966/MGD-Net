import torch
import torch.nn as nn
import torch.nn.functional as F
from .deform_conv_v2 import DeformConv2d
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import LayerNorm
from .SE_weight_module import SEWeightModule
from functools import partial
#from .basicnet import MutualNet
import math
# from memory_efficient_softmax import memory_efficient_softmax
from .deform_conv_v2 import DeformConv2d
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_
nonlinearity = partial(F.relu, inplace=True)

def memory_efficient_softmax(x, dim=-1, chunk_size=None, max_elems_per_chunk=30_000_000):
    """
    对 x 在 dim 维做 chunked softmax，避免一次性申请巨大的临时张量导致 OOM。
    - x: tensor
    - dim: softmax 的维度
    - chunk_size: 沿 dim 轴每次处理的元素数。如果 None 则自动选择：
        chunk_size = ceil(max_elems_per_chunk / (total_elems / size_dim))
      意味每 chunk 最多包含 max_elems_per_chunk 个元素（默认 30M ≈ 120MB for float32）
    - 返回: 与 torch.softmax(x, dim) 相同形状的 tensor
    """
    # 快路径
    if chunk_size is None:
        total_elems = x.numel()
        size_dim = x.size(dim)
        # 估计每 chunk 的元素数，使每 chunk 大小接近阈值
        est = max(1, math.ceil(max_elems_per_chunk / (total_elems / size_dim)))
        chunk_size = min(size_dim, est)

    if chunk_size >= x.size(dim):
        return torch.softmax(x, dim=dim)

    parts = []
    # 逐段用 narrow（避免复制多余维度）计算 softmax
    for start in range(0, x.size(dim), chunk_size):
        length = min(chunk_size, x.size(dim) - start)
        part = x.narrow(dim, start, length)
        parts.append(torch.softmax(part, dim=dim))
    return torch.cat(parts, dim=dim)
#添加多尺度注意力
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0, dilation=dilation),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class double_deform_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        attention = memory_efficient_softmax(energy, dim=-1, chunk_size=32)  # chunk_size 可调，试 8/16/32
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.se = SEWeightModule(self.channels_single)
        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        #p1 = self.p1(p1_input)
        #p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_input
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((self.se(p1_input)*p1_input, self.se(p2_dc)*p2_dc, self.se(p3_dc)*p3_dc, self.se(p4_dc)*p4_dc), 1))

        return ce

class GCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM, self).__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),#64channel
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            #NonLocalBlock(out_channels)
            SA_Block(64)
        ))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], kernel_size=1),#channel
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
            #print("global_context1", global_context[i].shape)
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)
        #print("global_context",global_context.shape)
        output = []
        for i in range(len(self.GCoutmodel)):
            output.append(self.GCoutmodel[i](global_context))

        return output
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels is None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # 平均池化路径
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        # 最大池化路径
        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        # 随机池化路径
        random_pool_out = self.stochastic_pool(x)
        random_out = self.fc2(self.activation(self.fc1(random_pool_out)))

        # 将三种池化的输出相加
        out = avg_out + max_out + random_out
        return self.sigmoid(out)

    def stochastic_pool(self, x):
        # 自适应随机池化：通过设置池化窗口大小为1，可以实现类似GAP和GMP的效果
        batch_size, channels, height, width = x.size()
        random_out = F.adaptive_avg_pool2d(x, (1, 1))  # 初始设定为GAP
        for b in range(batch_size):
            for c in range(channels):
                # 随机选择当前通道上的一个位置值作为池化输出
                rand_h = torch.randint(0, height, (1,))
                rand_w = torch.randint(0, width, (1,))
                random_out[b, c, 0, 0] = x[b, c, rand_h, rand_w]
        return random_out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0, dilation=dilation),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class double_deform_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))

class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out
class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))
class Conv2d_batchnorm_cab(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = CAB_o(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))
class GSI(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(GSI, self).__init__()
        # 确保 in_channels 和 reduction_ratio 是整数，且可以整除
        assert in_channels > 0, "in_channels 必须是正整数"
        assert reduction_ratio > 0, "reduction_ratio 必须是正整数"
        assert in_channels % reduction_ratio == 0, f"in_channels 必须能被 reduction_ratio {reduction_ratio} 整除"

        # 1x1 卷积层生成 Q, K, V
        self.conv_q = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 输出聚合卷积层
        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 输入张量形状: [B, C, H, W]

        # 通过1x1卷积生成 Q, K, V
        Q = self.conv_q(x)  # [B, C//r, H, W]
        K = self.conv_k(x)  # [B, C//r, H, W]
        V = self.conv_v(x)  # [B, C, H, W]

        # 融合 (Q 与 K 的点积)
        # 将 Q 和 K reshape 成 [B, C//r, H*W]，并执行矩阵乘法
        B, C_q, H, W = Q.shape
        Q_flatten = Q.view(B, C_q, -1)  # [B, C//r, H*W]
        K_flatten = K.view(B, C_q, -1)  # [B, C//r, H*W]

        # 计算注意力矩阵
        attention_map = torch.bmm(Q_flatten.permute(0, 2, 1), K_flatten)  # [B, H*W, H*W]
        attention_map = F.softmax(attention_map, dim=-1)  # 对注意力矩阵进行 softmax 归一化

        # 聚合 (将注意力应用于 V)
        V_flatten = V.view(B, -1, H * W)  # [B, C, H*W]
        aggregated = torch.bmm(V_flatten, attention_map)  # [B, C, H*W]

        # 将聚合后的结果 reshape 回原来的形状 [B, C, H, W]
        aggregated = aggregated.view(B, -1, H, W)

        # 通过聚合卷积层（比如1x1卷积），输出最终的特征图
        output = self.output_conv(aggregated)

        # 跳跃连接，加入原输入特征
        output = output + x

        return output


class CAB_o(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB_o, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        attention_weights = self.sigmoid(out)

        # 应用注意力权重到输入特征图
        return x * attention_weights

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out
class Conv2d_batchnorm_psa(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.psa = PSAModule(num_out_filters,num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.psa(self.activation(x))

class MLFC_psa(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels
        self.psa1 = PSAModule(in_filters1, in_filters1)
        self.psa2 = PSAModule(in_filters2, in_filters2)
        self.psa3 = PSAModule(in_filters3, in_filters3)
        self.psa4 = PSAModule(in_filters4, in_filters4)
        self.no_param_up = torch.nn.Upsample(scale_factor=2) # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(3 * in_filters1, in_filters1, (1, 1)))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(3 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(3 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1))
            )
            self.cnv_mrg4.append(Conv2d_batchnorm(3 * in_filters4, in_filters4, (1, 1)))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))


        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)


    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                (x1),
                                (self.no_param_up(x2)),
                                (self.no_param_up(self.no_param_up(x3))),
                                (self.no_param_up(self.no_param_up(self.no_param_up(x4)))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                (self.no_param_down(x1)),
                                (x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                (self.no_param_down(self.no_param_down(x1))),
                                (self.no_param_down(x2)),
                                (x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                (self.no_param_down(self.no_param_down(self.no_param_down(x1)))),
                                (self.no_param_down(self.no_param_down(x2))),
                               (self.no_param_down(x3)),
                               (x4),
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1,self.psa1(x1)], dim=2).view(batch_size, 3 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2,self.psa2(x2)], dim=2).view(batch_size, 3 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3,self.psa3(x3)], dim=2).view(batch_size, 3 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4,self.psa4(x4)], dim=2).view(batch_size, 3 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4

class MLFC(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(2 * in_filters1, in_filters1, (1, 1)))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1))
            )
            self.cnv_mrg4.append(Conv2d_batchnorm(2 * in_filters4, in_filters4, (1, 1)))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)


    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                (x1),
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                (x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                (x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                (x4),
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).view(batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).view(batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).view(batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).view(batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4
class MLFC_CBAM(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels
        self.cbam1 = CBAM(in_filters1)
        self.cbam2 = CBAM(in_filters2)
        self.cbam3 = CBAM(in_filters3)
        self.cbam4 = CBAM(in_filters4)

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(2 * in_filters1, in_filters1, (1, 1)))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1))
            )
            self.cnv_mrg4.append(Conv2d_batchnorm(2 * in_filters4, in_filters4, (1, 1)))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)


    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                self.cbam1(x1)+x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                self.cbam2(x2)+x2,
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                self.cbam3(x3)+x3,
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                self.cbam4(x4)+x4,
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).view(batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).view(batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).view(batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).view(batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4
class MLFC_new(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.gsi1 = GSI(in_filters1)
        self.gsi2 = GSI(in_filters2)
        self.gsi3 = GSI(in_filters3)
        self.gsi4 = GSI(in_filters4)
        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(2 * in_filters1, in_filters1, (1, 1)))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1))
            )
            self.cnv_mrg4.append(Conv2d_batchnorm(2 * in_filters4, in_filters4, (1, 1)))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)


    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                self.gsi1(x1),
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                self.gsi2(x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                self.gsi3(x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                self.gsi4(x4),
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).view(batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).view(batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).view(batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).view(batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x) * x
        # Spatial attention
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out, _ = torch.max(ca, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return sa * ca
class MLFA(torch.nn.Module):#加了CBAM，将卷积改为了空洞卷积
    """
    Implements Multi Level Feature Attention

    """

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4

        self.cbam1 = CBAM(in_filters1)
        self.cbam2 = CBAM(in_filters2)
        self.cbam3 = CBAM(in_filters3)
        self.cbam4 = CBAM(in_filters4)

        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                DoubleConv(self.in_filters, in_filters1,dilation=2)
            )
            self.cnv_mrg1.append(DoubleConv(2 * in_filters1, in_filters1,dilation=2))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                DoubleConv(self.in_filters, in_filters2,dilation=2)
            )
            self.cnv_mrg2.append(DoubleConv(2 * in_filters2, in_filters2,dilation=2))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                DoubleConv(self.in_filters, in_filters3,dilation=2)
            )
            self.cnv_mrg3.append(DoubleConv(2 * in_filters3, in_filters3,dilation=2))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                DoubleConv(self.in_filters, in_filters4,dilation=2)
            )
            self.cnv_mrg4.append(DoubleConv(2 * in_filters4, in_filters4,dilation=2))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)


    def forward(self, x1, x2, x3, x4):
        x1 = self.cbam1(x1)
        x2 = self.cbam2(x2)
        x3 = self.cbam3(x3)
        x4 = self.cbam4(x4)
        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                (x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                (x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                x4,
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).view(batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).view(batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).view(batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).view(batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )
        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4
