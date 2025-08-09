# -*- codeing = utf-8 -*-
# @Time :2023/10/30 19:52
# @Author:JHX
# @File : Resnet18_CM_smAR.py
# @Software: PyCharm


import torch
import torch.nn as nn

# from torchvision.models.utils import load_state_dict_from_url

from nets.Resnet18_backbone import Backbone_ResNet18
from nets.CM_module import CMLayer
from nets.Attention_module import ChannelAttention, SpatialAttention


class CIRNet_R18(nn.Module):
    """
    The implementation of "CIR-Net: Cross-Modality Interaction and Refinement for RGB-D Salient Object Detection"
    """

    def __init__(self, backbone='resnet18', norm_layer=nn.BatchNorm2d):
        #
        super(CIRNet_R18, self).__init__()
        (
            self.rgb_block1,
            self.rgb_block2,
            self.rgb_block3,
            self.rgb_block4,
            self.rgb_block5,
        ) = Backbone_ResNet18(pretrained=True)

        (
            self.fir_block1,
            self.fir_block2,
            self.fir_block3,
            self.fir_block4,
            self.fir_block5,
        ) = Backbone_ResNet18(pretrained=True)

        self.crossmodal1 = CMLayer()
        self.crossmodal2 = CMLayer()
        self.crossmodal3 = CMLayer()

        # self-modality attention refinement
        self.ca_rgb = ChannelAttention(512)
        self.ca_fir = ChannelAttention(512)

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_fir = SpatialAttention(kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1)

    def forward(self, rgb, fir):
        rgb = self.rgb_block1(rgb)
        fir = self.fir_block1(fir)

        rgb = self.rgb_block2(rgb)
        fir = self.fir_block2(fir)

        rgb_sum1, fir_sum1 = self.crossmodal1(rgb, fir)

        rgb = self.rgb_block3(rgb + rgb_sum1)
        fir = self.fir_block3(fir + fir_sum1)

        rgb_sum2, fir_sum2 = self.crossmodal2(rgb, fir)

        rgb = self.rgb_block4(rgb + rgb_sum2)
        fir = self.fir_block4(fir + fir_sum2)

        rgb_sum3, fir_sum3 = self.crossmodal3(rgb, fir)

        rgb = self.rgb_block5(rgb + rgb_sum3)
        fir = self.fir_block5(fir + fir_sum3)

        # self-modality attention refinement
        B, C, H, W = rgb.size()
        P = H * W

        rgb_SA = self.sa_rgb(rgb).view(B, -1, P)  # B * 1 * H * W
        fir_SA = self.sa_fir(fir).view(B, -1, P)

        rgb_CA = self.ca_rgb(rgb).view(B, C, -1)  # B * C * 1 * 1
        fir_CA = self.ca_fir(fir).view(B, C, -1)

        rgb_M = torch.bmm(rgb_CA, rgb_SA).view(B, C, H, W)  # torch.bmm(输入, mat2, *, out=None)
        fir_M = torch.bmm(fir_CA, fir_SA).view(B, C, H, W)  # 对输入和 mat2 中存储的矩阵执行批量矩阵-矩阵乘积。
        # input 和 mat2 必须是 3-D 张量，每个张量包含相同数量的矩阵。

        rgb_smAR = rgb * rgb_M + rgb
        fir_smAR = fir * fir_M + fir

        rgb = self.avgpool(rgb_smAR)
        fir = self.avgpool(fir_smAR)

        rgb = torch.flatten(rgb, 1)
        fir = torch.flatten(fir, 1)

        x = self.fc(torch.cat((rgb, fir), dim=1))

        return x

# 缺少一个展平的过程

def create_model(pretrained):
    model = CIRNet_R18()
    if pretrained:
        state_dict = torch.load('../weights/Fusion_Network_weights/the_best_Resnet18_AM_CM.pt')
        model.load_state_dict(state_dict)
    return model

#只需要切换这里的权重值，其他都不需要动，然后运行train_myself就可以训练全连接层了