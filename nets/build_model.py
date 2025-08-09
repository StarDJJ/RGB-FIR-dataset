# -*- codeing = utf-8 -*-
# @Time :2023/10/30 20:36
# @Author:JHX
# @File : build_model.py
# @Software: PyCharm

import torch
import torch.nn as nn

from nets.Resnet18_backbone import Backbone_ResNet18
from nets.CM_module import CMLayer
from nets.Attention_module import ChannelAttention, SpatialAttention
from nets.single_rgb_branch import Single_branch
from nets.Resnet18_CM_smAR import create_model


def build_branch_model(pretrained = False):

    overall_model = create_model(pretrained)
    # 访问整个模型的状态字典
    original_state_dict = overall_model.state_dict()

    #打印整个模型的权重参数
    # print(original_state_dict)

    branch_model = Single_branch()

    # 从原始模型的 state_dict 中提取 rgb_block 和 sa_rgb 部分的权重
    rgb_block_state_dict = {k: v for k, v in original_state_dict.items() if 'rgb_block' in k}
    sa_rgb_state_dict = {k: v for k, v in original_state_dict.items() if 'sa_rgb' in k}

    # 更新新模型的 state_dict，将 rgb_block 和 sa_rgb 的权重加载进来
    branch_model_state_dict = branch_model.state_dict()
    branch_model_state_dict.update(rgb_block_state_dict)
    branch_model_state_dict.update(sa_rgb_state_dict)

    # 加载更新后的 state_dict 到新模型中
    branch_model.load_state_dict(branch_model_state_dict)

    # print(branch_model)
    return branch_model

#得到有rgb权重值的rgb分支

'''
branch_model = build_branch_model(pretrained = True)

branch_model_state_dict = branch_model.state_dict()

#打印分割线
print("-----------------------------------------------------------------")
#打印出载入过权重的单分支网络权重参数
print(branch_model_state_dict)
'''