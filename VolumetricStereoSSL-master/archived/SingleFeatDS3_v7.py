# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:08:31 2020

@author: B016
"""

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        #        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(in_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        x = self.conv(x)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 is_conv2=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_conv2 = is_conv2
        self.deconv = deconv
        self.is_3d = is_3d

        if deconv and is_3d:
            ks = (3, 3, 3)
            ss = (2, 2, 2)
            pd = (1, 1, 1)
            op = (1, 1, 1)
            #            kernel = 3
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=ks,
                                   stride=ss, padding=pd, output_padding=op)
        elif deconv:
            #            kernel = 4
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=3,
                                   stride=2, padding=1, output_padding=1)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=3,
                                   stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        # print(x.size(), rem.size())
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        if self.is_conv2:
            x = self.conv2(x)
        return x

class HGFeature(nn.Module):
    def __init__(self):
        super(HGFeature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 64, kernel_size=3, stride=2, padding=1)
        # self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        # self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(64, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 64)
        # self.conv4b = Conv2x(96, 128)

        # self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(64, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)

        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class AsymThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # output = torch.sigmoid(*args, **kwargs)
        output = torch.where(input>=0, torch.ones_like(input), torch.zeros_like(input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_variables
        return grad_output * (1.00001 - output) * (0.00001 + output)

class CensusTransform(nn.Module):
    def __init__(self, wd=11):
        super(CensusTransform, self).__init__()
        self.wd = wd
        self.wd_hf = int(wd / 2)
        self.size = wd * wd -1

        self.offsets = []
        for i in range(1, self.wd_hf + 1):
            temp = [(u, v) for v in range(self.wd_hf - i, self.wd_hf + i + 1) for u in range(self.wd_hf - i, self.wd_hf + i + 1) if
                    (not u == self.wd_hf == v and not (u, v) in self.offsets)]
            self.offsets += temp

    def forward(self, x, attack=False):
        # assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            n, c, h, w = x.size()
            x = torch.mean(x, dim=1, keepdim=True)

            x_new = F.pad(x, (self.wd_hf, self.wd_hf, self.wd_hf, self.wd_hf), mode='reflect')
            # x_new = F.pad(x, (self.wd_hf, self.wd_hf, self.wd_hf, self.wd_hf), mode='constant', value=0)
            # census = x.new().resize_(n, self.size, h, w).zero_()

            if not attack:
                census = torch.zeros((n, self.size, h, w), dtype=torch.bool, device=x.device)
                for i, (u, v) in enumerate(self.offsets):
                    census[:, i, :, :] = x_new[:, 0, v:v + h, u:u + w] >= x[:, 0, :, :]
                    # census[:,i,:,:] = (x_new[:, 0, v:v + h, u:u + w] >= x[:, 0, :, :]).float()
            else:
                census = torch.zeros((n, self.size, h, w), dtype=torch.float, device=x.device)
                for i, (u, v) in enumerate(self.offsets):
                    # census[:, i, :, :] = AsymThreshold.apply((x_new[:, 0, v:v + h, u:u + w] - x[:, 0, :, :]))
                    census[:,i,:,:] = torch.sigmoid((x_new[:, 0, v:v + h, u:u + w] - x[:, 0, :, :])*100000)

        # census = census.contiguous()
        return census

class GetCostVolume(nn.Module):
    def __init__(self, maxdisp):
        super(GetCostVolume, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x, y, attack=False):
        # assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            #cost = x.new().resize_(num, 9, self.maxdisp, height, width).zero_()
            cost = torch.zeros((num, 9, self.maxdisp, height, width), dtype=torch.float, device=x.device)

            if not attack:
                for i in range(self.maxdisp):
                    if i > 0:
                        for k in range(3, 12):
                            idx = k * k
                            cost[:, k - 3, i, :, i:] = torch.mean((x[:, :idx, :, i:] != y[:, :idx, :, :-i]).float(), dim=1)
                    else:
                        for k in range(3, 12):
                            idx = k * k
                            cost[:, k - 3, i, :, i:] = torch.mean((x[:, :idx, :, :] != y[:, :idx, :, :]).float(), dim=1)
            else:
                for i in range(self.maxdisp):
                    if i > 0:
                        for k in range(3, 12):
                            idx = k * k
                            cost[:, k - 3, i, :, i:] = torch.mean(torch.abs(x[:, :idx, :, i:] - y[:, :idx, :, :-i]), dim=1)

                    else:
                        for k in range(3, 12):
                            idx = k * k
                            cost[:, k - 3, i, :, :] = torch.mean(torch.abs(x[:, :idx, :, :] - y[:, :idx, :, :]), dim=1)

            cost = cost.contiguous()
        return cost

class FeatToCV(nn.Module):
    def __init__(self, maxdisp):
        super(FeatToCV, self).__init__()
        self.maxdisp = int(maxdisp/3)

    def forward(self, x):
        # assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels, self.maxdisp, height, width).zero_()
            for i in range(self.maxdisp):
                if i > 0:
                    cost[:, :, i, :, i:] = x[:, :, :, i:]
                else:
                    cost[:, :, i, :, :] = x
            cost = cost.contiguous()
        return cost

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)), [1, self.maxdisp, 1, 1])).cuda(),
                            requires_grad=False)
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out


class Disp(nn.Module):

    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = torch.squeeze(x, 1)
        x = self.softmax(x)

        return self.disparity(x)


class CostRefine(nn.Module):
    def __init__(self, maxdisp=192, c=32):
        super(CostRefine, self).__init__()

        ks = (3, 3, 3)
        ss = (2, 2, 2)
        pd = (1, 1, 1)

        c2 = c  # *2
        c4 = c  # *4
        # c8 = c*8

        self.maxdisp = maxdisp

        self.conv_19 = BasicConv(c, c, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_20 = BasicConv(c, c, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_21 = BasicConv(c, c2, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_22 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_23 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_24 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_25 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_26 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_27 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_28 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_29 = BasicConv(c2, c2, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_30 = BasicConv(c2, c4, is_3d=True, kernel_size=ks, stride=ss, padding=pd)
        self.conv_31 = BasicConv(c4, c4, is_3d=True, kernel_size=ks, padding=pd)
        self.conv_32 = BasicConv(c4, c4, is_3d=True, kernel_size=ks, padding=pd)

        self.deconv_33 = Conv2x(c4, c2, deconv=True, is_3d=True, concat=False, is_conv2=False)
        self.deconv_34 = Conv2x(c2, c2, deconv=True, is_3d=True, concat=False, is_conv2=False)
        self.deconv_35 = Conv2x(c2, c2, deconv=True, is_3d=True, concat=False, is_conv2=False)
        self.deconv_36 = Conv2x(c2, c, deconv=True, is_3d=True, concat=False, is_conv2=False)

    def forward(self, x):
        bc = x
        bc = self.conv_19(bc)
        bc = self.conv_20(bc)
        rem_20 = bc

        x = self.conv_21(x)
        bc = self.conv_22(x)
        bc = self.conv_23(bc)
        rem_23 = bc

        x = self.conv_24(x)
        bc = self.conv_25(x)
        bc = self.conv_26(bc)
        rem_26 = bc

        x = self.conv_27(x)
        bc = self.conv_28(x)
        bc = self.conv_29(bc)
        rem_29 = bc

        x = self.conv_30(x)
        x = self.conv_31(x)
        x = self.conv_32(x)

        x = self.deconv_33(x, rem_29)
        x = self.deconv_34(x, rem_26)
        x = self.deconv_35(x, rem_23)
        x = self.deconv_36(x, rem_20)

        return x


class Model(nn.Module):
    def __init__(self, maxdisp=192, c=32, training=True):
        super(Model, self).__init__()
        self.training = training
        self.maxdisp = maxdisp
        self.feat_cv = FeatToCV(self.maxdisp)
        self.census = CensusTransform()
        self.cv = GetCostVolume(self.maxdisp)
        self.cost_refine_1 = CostRefine(self.maxdisp, c)
        self.cost_refine_2 = CostRefine(self.maxdisp, c)
        self.cost_refine_3 = CostRefine(self.maxdisp, c)
        self.disp = Disp(self.maxdisp)


        self.conv_start = BasicConv(9, c, is_3d=True, kernel_size=5, stride=3, padding=2)
        self.conv_start_2 = BasicConv(2*c, c, is_3d=True, kernel_size=3, padding=1)

        self.deconv_1 = BasicConv(c, 1, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=5, stride=3, padding=2,
                                output_padding=(2, 2, 2))
        self.deconv_2 = BasicConv(c, 1, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=5, stride=3, padding=2,
                                output_padding=(2, 2, 2))
        self.deconv_3 = BasicConv(c, 1, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=5, stride=3, padding=2,
                                output_padding=(2, 2, 2))
        # self.final_conv = BasicConv(c, 1, is_3d=True, bn=True, relu=True, kernel_size=1)
        self.feature = HGFeature()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, attack=False):
        feat = self.feature(x)
        f_cv = self.feat_cv(feat)

        x_c = self.census(x, attack=attack)
        y_c = self.census(y, attack=attack)
        cv = self.cv(x_c, y_c, attack=attack)

        cv = self.conv_start(cv)
        cv = torch.cat((cv, f_cv), 1)
        cv = self.conv_start_2(cv)

        cv1 = self.cost_refine_1(cv)
        cv2 = self.cost_refine_2(cv1)
        cv3 = self.cost_refine_3(cv2)
        # cv = self.final_conv(cv)
        # cv = F.interpolate(cv, [self.maxdisp, cv.size()[3] * 3, cv.size()[4] * 3], mode='trilinear', align_corners=False)

        cv3 = self.deconv_3(cv3)
        exp_disp_3 = self.disp(cv3)

        if self.training:
        # if True:
            cv1 = self.deconv_1(cv1)
            cv2 = self.deconv_2(cv2)
            exp_disp_1 = self.disp(cv1)
            exp_disp_2 = self.disp(cv2)
            return exp_disp_1, exp_disp_2, exp_disp_3

        return exp_disp_3