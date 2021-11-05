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


class LAM(nn.Module):
    def __init__(self, n_feats):
        super(LAM, self).__init__()

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, RG_num, channel, h, w = x.shape

        query = x.view(batch_size, RG_num, -1)
        key = x.view(batch_size, RG_num, -1).permute(0, 2, 1)
        value = x.view(batch_size, RG_num, -1)

        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, dim=-1, keepdim=True)[0].expand_as(energy) - energy  # zero centering
        attention = self.softmax(energy_new)

        out = torch.bmm(attention, value)
        out = out.view(batch_size, RG_num, channel, h, w)

        out = self.alpha * out + x
        out = out.view(batch_size, -1, h, w)
        return out


class CSAM(nn.Module):
    def __init__(self, in_channel):
        super(CSAM, self).__init__()

        self.in_channel = in_channel
        self.conv3d = nn.Conv3d(1, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, h, w = x.shape

        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv3d(out))

        query = out.view(batch_size, 1, -1)
        key = out.view(batch_size, 1, -1).permute(0, 2, 1)
        value = out.view(batch_size, 1, -1)
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention, value)
        out = out.view(batch_size, 1, channel, h, w)

        out = self.beta * out
        out = out.view(batch_size, -1, h, w)
        x = x * out + x

        return x


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


class HAN(nn.Module):
    def __init__(self, args,  rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), conv=common.default_conv):
        super(HAN, self).__init__()

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

        modules_body = [ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
                        for _ in range(self.n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [
            # common.Upsampler(conv, upsample_ratio, n_feats),
            conv(n_feats, 3, kernel_size)
        ]

        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.CSAM = CSAM(n_feats)
        self.LAM = LAM(n_feats)
        self.last_conv = nn.Conv2d(n_feats * self.n_resgroups, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        residual = x
        for name, midlayer in self.body._modules.items():
            residual = midlayer(residual)
            if name == '0':
                residual1 = residual.unsqueeze(1)
            elif name == str(self.n_resgroups):
                continue
            else:
                residual1 = torch.cat([residual.unsqueeze(1), residual1], 1)

        out1 = self.CSAM(residual)
        out2 = self.last_conv(self.LAM(residual1))
        out = torch.cat([out1, out2], dim=1)

        out = self.last(out) + x
        out = self.tail(out)
        # out = self.add_mean(out)

        return out


