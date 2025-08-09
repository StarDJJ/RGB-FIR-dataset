# -*- codeing = utf-8 -*-
# @Time :2023/10/30 19:51
# @Author:JHX
# @File : CM_module.py
# @Software: PyCharm

import torch
import torch.nn as nn

class CMLayer(nn.Module):
    def __init__(self):
        super(CMLayer, self).__init__()

    def forward(self, rgb, fir):
        part1 = rgb
        part2 = fir
        rgb_sum = (part1 + part2 + (part1 * part2))
        fir_sum = (part1 + part2 + (part1 * part2))
        return rgb_sum, fir_sum
