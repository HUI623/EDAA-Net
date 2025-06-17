#! /usr/bin/env/python3
# -*- coding=utf-8 -*-
'''
======================模块功能描述=========================    
       @File     : EDAA-Net.py
       @IDE      : PyCharm
       @Author   : Wanghui-BIT
       @Date     : 2024/7/16 19:52
       @Desc     : 
=========================================================   
'''

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import warnings
import torchvision
from models.EDAA_Net.otsu import *
from gcn.layers import GConv

warnings.filterwarnings(action='ignore')

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),  # False True
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):

        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# Edge and Density-Aware Attention Module
class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),  # 图像大小不变
            nn.BatchNorm2d(out_channel),
            # 防止过拟合
            # nn.Dropout2d(0.3),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.3),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out

class UpSize(nn.Module):
    """Upscaling then double conv"""

    # 上采样两种，转置卷积（空洞卷积）和插值法（线性和最邻近）
    def __init__(self, in_ch, out_ch):
        super(UpSize, self).__init__()

    def forward(self, x1, x2):
        upzise1 = F.interpolate(x1, scale_factor=2, mode='nearest')  # 输入图X的尺寸，插值后变成原来的2倍
        upzise2 = F.interpolate(x2, scale_factor=2, mode='nearest')  # 输入图X的尺寸，插值后变成原来的2倍
        out = torch.cat([upzise1, upzise2], dim=1)  # 32 + 32 -> 64

        return out

class gaborlayer(nn.Module):
    def __init__(self, in_dim):
        super(gaborlayer, self).__init__()
        self.layer = GConv(in_dim, out_channels=in_dim // 4, kernel_size=1, stride=1, M=4, nScale=1, bias=False,
                           expand=True)

    def forward(self, x):
        out = self.layer(x)
        return out

# DA Module
# Position Attention Module
class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.in_channel = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  # 在最后一维上进行归一化

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # ? B C W H
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B C N (N=W*H)-> B N C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B C N
        # 用于计算两个具有相同批次大小的三维张量的矩阵乘法  (B,n,m) * (B,m,p)=(B,n,p)
        energy = torch.bmm(proj_query, proj_key)  # B N C * B C N = B N N
        attention = self.softmax(energy)  # B N N 归一化
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B C N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B C N * B N N -> B C N
        out1 = out.view(m_batchsize, C, height, width)  # ? B C H W

        return out1


# 2. Density and Edge-Aware Module
# (1) Ostu Module
def otsu(img):
    return otsu_helper(img, categories=1)

def dootsu(img):
    _, c, _, _ = img.size()
    img = img.cpu().detach()
    channel = list(img.size())[1]
    batch = list(img.size())[0]
    imgfolder = img.chunk(batch, dim=0)  # 把一个tensor均匀分割成若干个小tensor
    chw_output = []
    for index in range(batch):
        bchw = imgfolder[index]
        chw = bchw.squeeze()
        chwfolder = chw.chunk(channel, dim=0)
        hw_output = []
        for i in range(channel):
            hw = chwfolder[i].squeeze()
            hw = np.transpose(hw.detach().numpy(), (0, 1))
            hw_otsu = otsu(hw)
            hw_otsu = torch.from_numpy(hw_otsu)
            hw_output.append(hw_otsu)
        chw_otsu = torch.stack(hw_output, dim=0)
        chw_output.append(chw_otsu)
    bchw_otsu = torch.stack(chw_output, dim=0).cuda()

    return bchw_otsu

# 双卷积
class PreActivateDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.double_conv(x)

# 下采样
class PreActivateResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.selayer = SELayer(out_channels)
        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.ch_avg(x)
        out = self.double_conv(x)
        out = out + identity
        out = self.selayer(out)
        return self.down_sample(out), out

# 上采样
class PreActivateResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreActivateResUpBlock, self).__init__()
        self.ch_avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = PreActivateDoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x) + self.ch_avg(x)


class EDAANet(nn.Module):
    def __init__(self,in_channel, out_classes):
        super(EDAANet, self).__init__()
        self.in_channel = in_channel
        self.out_classes = out_classes

        self.down_conv1 = PreActivateResBlock(in_channel, 64)
        self.down_conv2 = PreActivateResBlock(64, 128)
        self.down_conv3 = PreActivateResBlock(128, 256)
        self.down_conv4 = PreActivateResBlock(256, 512)

        self.ASSP = ASPP(512, 1024)
        # self.double_conv = PreActivateDoubleConv(512, 1024)

        self.up_conv4 = PreActivateResUpBlock(512 + 1024, 512)
        self.up_conv3 = PreActivateResUpBlock(256 + 512, 256)
        self.up_conv2 = PreActivateResUpBlock(128 + 256, 128)
        self.up_conv1 = PreActivateResUpBlock(128 + 64, 64)

        # EDAA Module
        # self-attention
        self.pred_1 = single_conv(ch_in=64, ch_out=32)  # 图像尺寸减小一半
        self.pred_2 = single_conv(ch_in=64, ch_out=32)

        self.pred_11 = UpSize(in_ch=32, out_ch=32)  # 图像尺寸增加一半，通道不变, 32 + 32 -> 64,

        self.gabor = gaborlayer(in_dim=32)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv_Atten1 = PAM_Module(32)
        self.conv_Atten2 = PAM_Module(32)

        # fusion module
        self.conv_fusion1 = DoubleConv(128, 64)  # 两个卷积层  96  128  160
        self.conv_fusion2 = nn.Conv2d(64, out_classes, kernel_size=1, stride=1, padding=0)  # v1

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)

        x = self.ASSP(x)
        # x = self.double_conv(x)

        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)

        pred_1 = self.pred_1(x)  # 32
        pred_2 = self.pred_2(x)  # 32

        # self-attention
        attention_higher = self.conv_Atten1(pred_1)  # 32
        out1 = dootsu(attention_higher)
        out2 = self.gamma * out1 + pred_1

        attention_lower = self.conv_Atten2(pred_2)
        out3 = self.gabor(attention_lower)
        out4 = self.gamma * out3 + pred_2

        y = self.pred_11(out2, out4)

        y1 = torch.cat((y, x), dim=1)  # C = 64 + 64

        # fusion module
        y2 = self.conv_fusion1(y1)  # 128 -> 64
        pred = self.conv_fusion2(y2)  # 64 -> 3

        return pred


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 1, 256, 256).to(device)   # [16, 1, 128, 128]
    net = EDAANet(1, 2).to(device)
    print(net(x).shape)   # torch.Size([16, 2, 128, 128])  torch.Size([4, 2, 256, 256])
