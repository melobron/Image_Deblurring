import torch
import torch.nn as nn
import torch.nn.functional as F

from models import common


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        body = []
        for i in range(2):
            body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                body.append(act)

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        x = x + res * self.res_scale
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.u = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, channel, h, w = x.shape

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        u_x = self.u(x).view(batch_size, self.inter_channels, -1)
        u_x = u_x.permute(0, 2, 1)

        v_x = self.v(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(u_x, v_x)
        f = F.softmax(f, dim=1)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, h, w)

        W_y = self.W(y)
        z = W_y + x

        return z


class TrunkBranch(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(TrunkBranch, self).__init__()

        body = []
        modules_body = []
        for i in range(2):
            modules_body.append(
                ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)


class MaskBranch(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(MaskBranch, self).__init__()

        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, 2, 1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat, n_feat, 6, 2, 2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        MB_1by1conv = []
        MB_1by1conv.append(nn.Conv2d(n_feat, n_feat, 1, 1, 0, bias=bias))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1by1conv = nn.Sequential(*MB_1by1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1by1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)
        return mx


class NLMaskBranch(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLMaskBranch, self).__init__()

        MB_RB1 = []
        MB_RB1.append(NonLocalBlock(n_feat, n_feat // 2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, 2, 1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat, n_feat, 6, 2, 2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        MB_1by1conv = []
        MB_1by1conv.append(nn.Conv2d(n_feat, n_feat, 1, 1, 0, bias=bias))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1by1conv = nn.Sequential(*MB_1by1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1by1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)
        return mx


class ResAttBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttBlock, self).__init__()

        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        RA_MB = []
        RA_MB.append(MaskBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, x):
        RA_RB1_x = self.RA_RB1(x)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


class NLResAttBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttBlock, self).__init__()

        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=bias, bn=bn, act=act, res_scale=res_scale))

        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        RA_MB = []
        RA_MB.append(NLMaskBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, x):
        RA_RB1_x = self.RA_RB1(x)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


class RNAN(nn.Module):
    def __init__(self, args, rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), conv=common.default_conv):
        super(RNAN, self).__init__()

        n_feats = args.n_feats
        res_scale = args.res_scale
        upsample_ratio = args.upsample_ratio
        RAB_num = args.RAB_num
        RNAN_version = args.RNAN_version  # 1: (T,T), 2: (T,F), 3: (F,T), 4: (F,F)

        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)

        modules_head = [conv(3, n_feats, kernel_size)]

        modules_body = []
        if RNAN_version == 1 or RNAN_version == 2: modules_body.append(NLResAttBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale))
        for _ in range(RAB_num):
            modules_body.append(ResAttBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale))
        if RNAN_version == 1 or RNAN_version == 3: modules_body.append(NLResAttBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale))
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
        x = x + residual

        x = self.tail(x)
        # x = self.add_mean(x)

        return x
