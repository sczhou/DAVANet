#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

from models.submodules import *

class DispNet_Bi(nn.Module):
    def __init__(self):
        super(DispNet_Bi, self).__init__()
        # encoder
        ks = 3
        self.conv0 = conv(6, 48, kernel_size=ks, stride=1)

        self.conv1_1 = conv(48, 48, kernel_size=ks, stride=2)
        self.conv1_2 = conv(48, 48, kernel_size=ks, stride=1)

        self.conv2_1 = conv(48, 96, kernel_size=ks, stride=2)
        self.conv2_2 = conv(96, 96, kernel_size=ks, stride=1)

        self.conv3_1 = conv(96, 128, kernel_size=ks, stride=2)
        self.conv3_2 = conv(128, 128, kernel_size=ks, stride=1)

        self.conv4_1 = resnet_block(128, kernel_size=ks)
        self.conv4_2 = resnet_block(128, kernel_size=ks)

        self.convd_1 = resnet_block(128, kernel_size=ks, dilation=[2, 1])
        self.convd_2 = ms_dilate_block(128, kernel_size=ks, dilation=[1, 2, 3, 4])

        # decoder
        self.upconvd_i = conv(128, 128, kernel_size=ks, stride=1)
        self.dispd = predict_disp_bi(128)

        self.upconv3 = conv(128, 128, kernel_size=ks, stride=1)
        self.upconv3_i = conv(258, 128, kernel_size=ks, stride=1)
        self.upconv3_f = conv(128, 128, kernel_size=ks, stride=1)
        self.disp3 = predict_disp_bi(128)

        self.updisp3 = up_disp_bi()
        self.upconv2 = upconv(128, 96)
        self.upconv2_i = conv(194, 96, kernel_size=ks, stride=1)
        self.upconv2_f = conv(96, 96, kernel_size=ks, stride=1)
        self.disp2 = predict_disp_bi(96)

        self.updisp2 = up_disp_bi()
        self.upconv1 = upconv(96, 48)
        self.upconv1_i = conv(50, 48, kernel_size=ks, stride=1)
        self.upconv1_f = conv(48, 48, kernel_size=ks, stride=1)
        self.disp1 = predict_disp_bi(48)

        self.updisp1 = up_disp_bi()
        self.upconv0 = upconv(48, 32)
        self.upconv0_i = conv(34, 32, kernel_size=ks, stride=1)
        self.upconv0_f = conv(32, 32, kernel_size=ks, stride=1)
        self.disp0 = predict_disp_bi(32)

    def forward(self, x):
        # encoder
        conv0 = self.conv0(x)
        conv1 = self.conv1_2(self.conv1_1(conv0))
        conv2 = self.conv2_2(self.conv2_1(conv1))
        conv3 = self.conv3_2(self.conv3_1(conv2))
        conv4 = self.conv4_2(self.conv4_1(conv3))
        convd = self.convd_2(self.convd_1(conv4))

        # decoder
        upconvd_i = self.upconvd_i(convd)
        disp4 = self.dispd(upconvd_i)

        upconv3 = self.upconv3(upconvd_i)
        cat3 = torch.cat([conv3, upconv3, disp4], 1)
        upconv3_i = self.upconv3_f(self.upconv3_i(cat3))
        disp3 = self.disp3(upconv3_i) + disp4

        updisp3 = self.updisp3(disp3)
        upconv2 = self.upconv2(upconv3_i)
        cat2 = torch.cat([conv2, upconv2, updisp3], 1)
        upconv2_i = self.upconv2_f(self.upconv2_i(cat2))
        disp2 = self.disp2(upconv2_i) + updisp3

        updisp2 = self.updisp2(disp2)
        upconv1 = self.upconv1(upconv2_i)
        cat1 = torch.cat([upconv1, updisp2], 1)
        upconv1_i = self.upconv1_f(self.upconv1_i(cat1))
        disp1 = self.disp1(upconv1_i) + updisp2

        updisp1 = self.updisp1(disp1)
        upconv0 = self.upconv0(upconv1_i)
        cat0 = torch.cat([upconv0, updisp1], 1)
        upconv0_i = self.upconv0_f(self.upconv0_i(cat0))
        disp0 = self.disp0(upconv0_i) + updisp1

        # if self.training:
        #     return disp0, disp1, disp2, disp3, disp4
        # else:
        #     return disp0

        if self.training:
            return disp0, disp1, disp2, disp3, disp4, upconv0_i
        else:
            return disp0, upconv0_i