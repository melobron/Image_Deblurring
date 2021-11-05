import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, act=nn.ReLU(inplace=True), transpose=False):
        super(BasicBlock, self).__init__()
        if bias and bn:
            bias = False

        padding = kernel_size // 2
        layer_list = []

        if transpose:
            padding = kernel_size // 2 - 1
            layer_list.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        else:
            layer_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=globals()))
        if bn:
            layer_list.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            layer_list.append(act)

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1):
        super(ResidualBlock, self).__init__()

        self.res_scale = res_scale

        layer_list = []
        layer_list.append(conv(n_feats, n_feats, kernel_size, bias=bias))

        for _ in range(2):
            layer_list.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                layer_list.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                layer_list.append(act)

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.res_scale * self.layers(x) + x


class Upsampler(nn.Module):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()

        layer_list = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layer_list.append(conv(n_feats, 4 * n_feats, 3, bias))
                layer_list.append(nn.PixelShuffle(2))
                if bn:
                    layer_list.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    layer_list.append(nn.ReLU(True))
                elif act == 'prelu':
                    layer_list.append(nn.PReLU(n_feats))

        elif scale == 3:
            layer_list.append(conv(n_feats, 9 * n_feats, 3, bias))
            layer_list.append(nn.PixelShuffle(3))
            if bn:
                layer_list.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                layer_list.append(nn.ReLU(True))
            elif act == 'prelu':
                layer_list.append(nn.PReLU(n_feats))

        elif scale == 5:
            layer_list.append(conv(n_feats, 25 * n_feats, 3, bias))
            layer_list.append(nn.PixelShuffle(5))
            if bn:
                layer_list.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                layer_list.append(nn.ReLU(True))
            elif act == 'prelu':
                layer_list.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)

