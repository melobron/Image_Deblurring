import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import *


class RBs(nn.Module):
    def __init__(self, n_feats, num_res=8):
        super(RBs, self).__init__()

        layer_list = [ResidualBlock(default_conv, n_feats, 3) for _ in range(8)]
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AFF, self).__init__()

        layer_list = [
            BasicBlock(in_channels, out_channels, kernel_size=1, stride=1, act=nn.ReLU(inplace=True)),
            BasicBlock(out_channels, out_channels, kernel_size=3, stride=1, act=None)
        ]
        self.layer = nn.Sequential(*layer_list)

    def forward(self, EB1_out, EB2_out, EB3_out):
        x = torch.cat([EB1_out, EB2_out, EB3_out], dim=1)
        return self.layer(x)


class SCM(nn.Module):
    def __init__(self, out_channels):
        super(SCM, self).__init__()

        self.body = nn.Sequential(
            BasicBlock(3, out_channels // 4, kernel_size=3, stride=1, act=nn.ReLU(inplace=True)),
            BasicBlock(out_channels // 4, out_channels // 2, kernel_size=1, stride=1, act=nn.ReLU(inplace=True)),
            BasicBlock(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, act=nn.ReLU(inplace=True)),
            BasicBlock(out_channels // 2, out_channels-3, kernel_size=1, stride=1, act=nn.ReLU(inplace=True))
        )

        self.conv = BasicBlock(out_channels, out_channels, kernel_size=1, stride=1, act=None)

    def forward(self, x):
        x = torch.cat([x, self.body(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, n_feats):
        super(FAM, self).__init__()

        self.conv = BasicBlock(n_feats, n_feats, kernel_size=3, stride=1, act=None)

    def forward(self, EB_out, SCM_out):
        x = EB_out * SCM_out
        out = EB_out + self.conv(x)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        n_feats = 32

        self.Encoder = nn.ModuleList([
            RBs(n_feats, num_res),
            RBs(n_feats*2, num_res),
            RBs(n_feats*4, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicBlock(3, n_feats, kernel_size=3, stride=1, act=nn.ReLU(inplace=True)),
            BasicBlock(n_feats, n_feats*2, kernel_size=3, stride=2, act=nn.ReLU(inplace=True)),
            BasicBlock(n_feats*2, n_feats*4, kernel_size=3, stride=2, act=nn.ReLU(inplace=True)),
            BasicBlock(n_feats*4, n_feats*2, kernel_size=4, stride=2, act=nn.ReLU(inplace=True), transpose=True),
            BasicBlock(n_feats*2, n_feats, kernel_size=4, stride=2, act=nn.ReLU(inplace=True), transpose=True),
            BasicBlock(n_feats, 3, kernel_size=3, stride=1, act=None)
        ])

        self.Decoder = nn.ModuleList([
            RBs(n_feats*4, num_res),
            RBs(n_feats*2, num_res),
            RBs(n_feats, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicBlock(n_feats*4, n_feats*2, kernel_size=1, stride=1, act=nn.ReLU(inplace=True)),
            BasicBlock(n_feats*2, n_feats, kernel_size=1, stride=1, act=nn.ReLU(inplace=True))
        ])

        self.ConvsOut = nn.ModuleList([
            BasicBlock(n_feats*4, 3, kernel_size=3, stride=1, act=nn.ReLU(inplace=True)),
            BasicBlock(n_feats*2, 3, kernel_size=3, stride=1, act=nn.ReLU(inplace=True))
        ])

        self.AFFs = nn.ModuleList([
            AFF(n_feats*7, n_feats),
            AFF(n_feats*7, n_feats*2)
        ])

        self.FAM2 = FAM(n_feats*2)
        self.SCM2 = SCM(n_feats*2)
        self.FAM3 = FAM(n_feats*4)
        self.SCM3 = SCM(n_feats*4)

    def forward(self, x):
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
        z2 = self.SCM2(x2)
        z3 = self.SCM3(x3)

        outputs = []

        x1 = self.feat_extract[0](x)
        res1 = self.Encoder[0](x1)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM3(z, z3)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z32 = F.interpolate(z, scale_factor=2)
        z31 = F.interpolate(z32, scale_factor=2)

        res1 = self.AFFs[0](res1, z21, z31)
        res2 = self.AFFs[1](z12, res2, z32)

        z = self.Decoder[0](z)
        s3 = self.ConvsOut[0](z)
        outputs.append(s3 + x3)
        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        s2 = self.ConvsOut[1](z)
        outputs.append(s2 + x2)

        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        s1 = self.feat_extract[5](z)
        outputs.append(s1 + x)

        return outputs
