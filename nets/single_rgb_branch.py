# -*- codeing = utf-8 -*-
# @Time :2023/10/30 20:02
# @Author:JHX
# @File : single_rgb_branch.py
# @Software: PyCharm

import torch
import torch.nn as nn

from nets.Resnet18_backbone import Backbone_ResNet18
from nets.CM_module import CMLayer
from nets.Attention_module import ChannelAttention, SpatialAttention

from nets.Resnet18_CM_smAR import create_model


class Single_branch(nn.Module):
    """
    The implementation of "CIR-Net: Cross-Modality Interaction and Refinement for RGB-D Salient Object Detection"
    """

    def __init__(self, backbone='resnet18', norm_layer=nn.BatchNorm2d):
        #
        super(Single_branch, self).__init__()
        (
            self.rgb_block1,
            self.rgb_block2,
            self.rgb_block3,
            self.rgb_block4,
            self.rgb_block5,
        ) = Backbone_ResNet18(pretrained=True)

        # self-modality attention refinement
        self.ca_rgb = ChannelAttention(512)

        self.sa_rgb = SpatialAttention(kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, rgb):
        rgb = self.rgb_block1(rgb)

        rgb = self.rgb_block2(rgb)

        rgb = self.rgb_block3(rgb)

        rgb = self.rgb_block4(rgb)

        rgb = self.rgb_block5(rgb)

        # self-modality attention refinement
        B, C, H, W = rgb.size()
        P = H * W

        rgb_SA = self.sa_rgb(rgb).view(B, -1, P)  # B * 1 * H * W

        rgb_CA = self.ca_rgb(rgb).view(B, C, -1)  # B * C * 1 * 1

        rgb_M = torch.bmm(rgb_CA, rgb_SA).view(B, C, H, W)  # torch.bmm(输入, mat2, *, out=None)
        # input 和 mat2 必须是 3-D 张量，每个张量包含相同数量的矩阵。

        rgb_smAR = rgb * rgb_M + rgb

        rgb = self.avgpool(rgb_smAR)

        rgb = torch.flatten(rgb, 1)


        x = self.fc(rgb)

        return x

# 缺少一个展平的过程

# model = Single_branch()
# # print(model)
#
# feature_extractor = nn.Sequential(*list(model.children())[:-1])
# print(feature_extractor)

if __name__ == '__main__':
    from torchsummary import summary

    # Create an instance of the network
    model = Single_branch()

    # Move the model to 'cuda' if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print the summary
    summary(model, input_size=(3, 224, 224))
