import torch
import torch.nn as nn

from nets.build_model import build_branch_model

'''
def get_img_output_length(width, height):
    out_width = width
    out_height = height
    for i in range(4):
        out_width = (out_width + 1) // 2
        out_height = (out_height + 1) // 2

    return out_width * out_height
'''

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1, 1]
        padding = [3, 1, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height)


class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.branch = build_branch_model(pretrained)

        self.feature_extractor = nn.Sequential(*list(self.branch.children())[:-4])
        # del self.branch.avgpool
        # del self.branch.fc

        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        # -------------------------#
        #   相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)

        return x
