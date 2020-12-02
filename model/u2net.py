from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .squeeze_and_excitation import (ChannelSELayer, ChannelSpatialSELayer,
                                     SpatialSELayer)


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate,
                                 dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class SEConvBNReLU(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, se_type="CSSE"):
        super(SEConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate,
                              dilation=1*dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if se_type == "none" or not se_type:
            self.se = nn.Identity()
        elif se_type.lower() == "CSSE".lower():
            self.se = ChannelSpatialSELayer(out_ch)
        elif se_type.lower() == "SSE".lower():
            self.se = SpatialSELayer(out_ch)
        elif se_type.lower() == "CSE".lower():
            self.se = ChannelSELayer(out_ch)
        else:
            raise ValueError(f"Unknown se_type: {se_type}")

    def forward(self, x):

        hx = x
        xout = self.se(self.relu(self.bn(self.conv(hx))))

        return xout


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:],
                        mode='bilinear',
                        align_corners=False)
    return src


class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, se_type=None):
        super(RSU7, self).__init__()

        basic_block = partial(SEConvBNReLU, se_type=se_type) if se_type not in [
            None, "none", ""] else REBNCONV

        self.rebnconvin = basic_block(in_ch, out_ch, dirate=1)

        self.rebnconv1 = basic_block(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = basic_block(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = basic_block(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = basic_block(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, se_type=None):
        super(RSU6, self).__init__()

        basic_block = partial(SEConvBNReLU, se_type=se_type) if se_type not in [
            None, "none", ""] else REBNCONV

        self.rebnconvin = basic_block(in_ch, out_ch, dirate=1)

        self.rebnconv1 = basic_block(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = basic_block(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = basic_block(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = basic_block(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, se_type=None):
        super(RSU5, self).__init__()

        basic_block = partial(SEConvBNReLU, se_type=se_type) if se_type not in [
            None, "none", ""] else REBNCONV

        self.rebnconvin = basic_block(in_ch, out_ch, dirate=1)

        self.rebnconv1 = basic_block(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = basic_block(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = basic_block(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = basic_block(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, se_type=None):
        super(RSU4, self).__init__()

        basic_block = partial(SEConvBNReLU, se_type=se_type) if se_type not in [
            None, "none", ""] else REBNCONV

        self.rebnconvin = basic_block(in_ch, out_ch, dirate=1)

        self.rebnconv1 = basic_block(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = basic_block(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = basic_block(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = basic_block(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = basic_block(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = basic_block(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, se_type=None):
        super(RSU4F, self).__init__()

        basic_block = partial(SEConvBNReLU, se_type=se_type) if se_type not in [
            None, "none", ""] else REBNCONV

        self.rebnconvin = basic_block(in_ch, out_ch, dirate=1)

        self.rebnconv1 = basic_block(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = basic_block(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = basic_block(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = basic_block(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = basic_block(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = basic_block(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = basic_block(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, se_type=None):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64, se_type=se_type)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128, se_type=se_type)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256, se_type=se_type)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512, se_type=se_type)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512, se_type=se_type)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512, se_type=se_type)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512, se_type=se_type)
        self.stage4d = RSU4(1024, 128, 256, se_type=se_type)
        self.stage3d = RSU5(512, 64, 128, se_type=se_type)
        self.stage2d = RSU6(256, 32, 64, se_type=se_type)
        self.stage1d = RSU7(128, 16, 64, se_type=se_type)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, se_type=None):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64, se_type=se_type)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64, se_type=se_type)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64, se_type=se_type)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64, se_type=se_type)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64, se_type=se_type)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64, se_type=se_type)

        # decoder
        self.stage5d = RSU4F(128, 16, 64, se_type=se_type)
        self.stage4d = RSU4(128, 16, 64, se_type=se_type)
        self.stage3d = RSU5(128, 16, 64, se_type=se_type)
        self.stage2d = RSU6(128, 16, 64, se_type=se_type)
        self.stage1d = RSU7(128, 16, 64, se_type=se_type)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


class U2NET_Encoder(nn.Module):

    def __init__(self, in_ch=3, output_channels=[64, 128, 256, 512, 512, 512], se_type=None):
        super(U2NET_Encoder, self).__init__()

        self.stage1 = RSU7(in_ch, 32, output_channels[0], se_type=se_type)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(output_channels[0],
                           output_channels[0]//2,
                           output_channels[1],
                           se_type=se_type)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(output_channels[1],
                           output_channels[1]//2,
                           output_channels[2],
                           se_type=se_type)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(output_channels[2],
                           output_channels[2]//2,
                           output_channels[3],
                           se_type=se_type)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(output_channels[3],
                            output_channels[3]//2,
                            output_channels[4],
                            se_type=se_type)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(output_channels[4],
                            output_channels[4]//2,
                            output_channels[5],
                            se_type=se_type)

    def forward(self, x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        # hx6up = _upsample_like(hx6, hx5)

        x = [hx1, hx2, hx3, hx4, hx5, hx6]
        # for t in x:
        # print(f"x.shape: {t.shape}")

        return x


class U2NET_Decoder(nn.Module):

    def __init__(self,
                 in_channels=[512, 512, 512, 256, 128, 64],
                 mid_channels=[256, 128, 64, 32, 16],
                 channels=[512, 256, 128, 64, 64],
                 out_ch=1,
                 se_type=None):
        super(U2NET_Decoder, self).__init__()

        self.stage5d = RSU4F(in_channels[0] + in_channels[1], mid_channels[0], channels[0],
                             se_type=se_type)
        self.stage4d = RSU4(in_channels[2] + channels[0], mid_channels[1], channels[1],
                            se_type=se_type)
        self.stage3d = RSU5(in_channels[3] + channels[1], mid_channels[2], channels[2],
                            se_type=se_type)
        self.stage2d = RSU6(in_channels[4] + channels[2], mid_channels[3], channels[3],
                            se_type=se_type)
        self.stage1d = RSU7(in_channels[5] + channels[3], mid_channels[4], channels[4],
                            se_type=se_type)

        self.side1 = nn.Conv2d(channels[4], out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(channels[3], out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(channels[2], out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(channels[1], out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(channels[0], out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(in_channels[0], out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x, return_side_outputs=True):

        hx1, hx2, hx3, hx4, hx5, hx6 = x

        hx6up = _upsample_like(hx6, hx5)
        # print(f"hx6up shape: {hx6up.shape}")

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        # print(f"hx5dup shape: {hx5dup.shape}")

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        # print(f"hx4dup shape: {hx4dup.shape}")

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        # print(f"hx3dup shape: {hx3dup.shape}")

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        # print(f"hx2dup shape: {hx2dup.shape}")

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        if return_side_outputs:
            return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)
            # return d0, d1, d2, d3, d4, d5, d6
        else:
            return torch.sigmoid(d0)
            # return d0


class U2NET_Whole(nn.Module):

    def __init__(self, in_ch=3, out_ch=1,
                 encoder_channels=[64, 128, 256, 512, 512, 512],
                 decoder_mid_channels=[256, 128, 64, 32, 16],
                 decoder_channels=[512, 256, 128, 64, 64],
                 se_type=None
                 ):

        super(U2NET_Whole, self).__init__()
        self.encoder_channels = encoder_channels
        self.decoder_mid_channels = decoder_mid_channels
        self.decoder_channels = decoder_channels
        self.se_type = se_type

        self.encoder = U2NET_Encoder(in_ch,
                                     output_channels=encoder_channels,
                                     se_type=se_type)
        self.decoder = U2NET_Decoder(out_ch=out_ch,
                                     in_channels=encoder_channels[::-1],
                                     mid_channels=decoder_mid_channels,
                                     channels=decoder_channels,
                                     se_type=se_type)

    def forward(self, x, return_side_outputs=True):
        x = self.encoder(x)
        x = self.decoder(x, return_side_outputs=return_side_outputs)
        return x


def u2net_standard(se_type=None):
    net = U2NET(3, 1, se_type)
    return net


def u2net_portable(se_type=None):
    net = U2NETP(3, 1, se_type)
    return net


def u2net_heavy(se_type=None):
    net = U2NET_Whole(3, 1,
                      encoder_channels=[96, 192, 384, 768, 768, 768],
                      decoder_mid_channels=[384, 192, 96, 48, 48],
                      decoder_channels=[768, 384, 192, 96, 96],
                      se_type=se_type)
    return net


if __name__ == "__main__":
    from torchsummary import summary

    model = U2NET(3, 1, se_type="csse")
    summary(model, (torch.rand(4, 3, 320, 320),))
