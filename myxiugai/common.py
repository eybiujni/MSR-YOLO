import torch
import torch.nn as nn

# final
class LRMAMR(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LRMAMR, self).__init__()
        mid_channels = in_channels // 2

        # 4条特征分支保持不变
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, mid_channels, 1),
            BasicConv(mid_channels, mid_channels, 3, stride, 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, mid_channels, 1),
            BasicConv(mid_channels, mid_channels, (1, 3), stride, (0, 1)),
            BasicConv(mid_channels, mid_channels, (3, 1), 1, (1, 0), relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_channels, mid_channels, 1),
            BasicConv(mid_channels, mid_channels, (3, 1), stride, (1, 0)),
            BasicConv(mid_channels, mid_channels, (1, 3), 1, (0, 1), relu=False)
        )
        self.branch4 = nn.Sequential(
            BasicConv(in_channels, mid_channels, 1),
            BasicConv(mid_channels, mid_channels, 3, stride, 3, dilation=3, relu=False)
        )

        # 高分辨率分支
        self.high_res = nn.Conv2d(in_channels, mid_channels // 2, 1)
        cat_channels = 4 * mid_channels + mid_channels // 2

        # --- 核心改进：极其轻量化的权重模块 (参数量几乎忽略不计) ---
        # 使用 3x3 深度可分离卷积代替原来的空间注意力一部分，增加特征交互
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # --- 核心改进：将输出卷积改为带有更强交互能力的结构 ---
        self.conv_out = nn.Conv2d(cat_channels, out_channels, 1, 1, 0, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act_out = nn.SiLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # high_res = self.high_res(x)
        high_res = self.high_res(x) + 0.1 * x[:, :self.high_res.out_channels, :, :]

        out = torch.cat([b1, b2, b3, b4, high_res], dim=1)

        # 改进 1：空间注意力使用残差增强 (无需额外参数)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_weight = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))

        # 使用残差连接：保留原始特征的同时，强化显著区域
        out = out + out * spatial_weight

        # 改进 2：在最后的映射前加入简单的通道缩放 (Global Context)
        # 这是一个极简的注意力，计算每个通道的均值并乘回去
        scale = torch.mean(out, dim=(2, 3), keepdim=True)
        out = out * torch.sigmoid(scale)

        out = self.bn_out(self.conv_out(out))
        out += identity
        return self.act_out(out)

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, relu=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # 改进 1：将 ReLU 统一改为 SiLU (YOLOv8 官方标准)
        self.act = nn.SiLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

