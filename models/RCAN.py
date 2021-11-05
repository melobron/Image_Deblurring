import torch
import torch.nn as nn

from models import common


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel//16, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//16, channel, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.avg_pool(x)
        residual = self.body(residual)
        return x * residual


class RCAB(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()

        self.res_scale = res_scale

        layer_list = []
        for i in range(2):
            layer_list.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: layer_list.append(nn.BatchNorm2d(n_feats))
            if i == 0: layer_list.append(act)
        layer_list.append(CALayer(n_feats, reduction))
        self.body = nn.Sequential(*layer_list)

    def forward(self, x):
        residual = self.body(x)
        return residual + x


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()

        layer_list = [RCAB(conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
                      for _ in range(n_resblocks)]
        layer_list.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*layer_list)

    def forward(self, x):
        res = self.body(x)
        return res + x


class RCAN(nn.Module):
    def __init__(self, args, rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), conv=common.default_conv):
        super(RCAN, self).__init__()

        self.n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        upsample_ratio = args.upsample_ratio
        act = nn.ReLU(True)

        rgb_mean = rgb_mean
        rgb_std = rgb_std

        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)

        modules_head = [conv(3, n_feats, kernel_size)]

        modules_body = [ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale,
                                      n_resblocks=n_resblocks)
                        for _ in range(self.n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [
            # common.Upsampler(conv, upsample_ratio, n_feats),
            conv(n_feats, 3, kernel_size)
        ]

        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        residual = self.body(x)
        x += residual

        x = self.tail(x)
        # x = self.add_mean(x)

        return x


