#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

from models.submodules import *
from utils.network_utils import *
from config import cfg

class StereoDeblurNet(nn.Module):
    def __init__(self):
        super(StereoDeblurNet, self).__init__()
        # encoder
        ks = 3
        self.conv1_1 = conv(3, 32, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(32, kernel_size=ks)
        self.conv1_3 = resnet_block(32, kernel_size=ks)
        self.conv1_4 = resnet_block(32, kernel_size=ks)

        self.conv2_1 = conv(32, 64, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(64, kernel_size=ks)
        self.conv2_3 = resnet_block(64, kernel_size=ks)
        self.conv2_4 = resnet_block(64, kernel_size=ks)

        self.conv3_1 = conv(64, 128, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(128, kernel_size=ks)
        self.conv3_3 = resnet_block(128, kernel_size=ks)
        self.conv3_4 = resnet_block(128, kernel_size=ks)

        dilation = [1,2,3,4]
        self.convd_1 = resnet_block(128, kernel_size=ks, dilation = [2, 1])
        self.convd_2 = resnet_block(128, kernel_size=ks, dilation = [3, 1])
        self.convd_3 = ms_dilate_block(128, kernel_size=ks, dilation = dilation)

        self.gatenet = gatenet()

        self.depth_sense_l = depth_sense(33, 32, kernel_size=ks)
        self.depth_sense_r = depth_sense(33, 32, kernel_size=ks)

        # decoder
        self.upconv3_i = conv(288, 128, kernel_size=ks,stride=1)
        self.upconv3_3 = resnet_block(128, kernel_size=ks)
        self.upconv3_2 = resnet_block(128, kernel_size=ks)
        self.upconv3_1 = resnet_block(128, kernel_size=ks)

        self.upconv2_u = upconv(128, 64)
        self.upconv2_i = conv(128, 64, kernel_size=ks,stride=1)
        self.upconv2_3 = resnet_block(64, kernel_size=ks)
        self.upconv2_2 = resnet_block(64, kernel_size=ks)
        self.upconv2_1 = resnet_block(64, kernel_size=ks)

        self.upconv1_u = upconv(64, 32)
        self.upconv1_i = conv(64, 32, kernel_size=ks,stride=1)
        self.upconv1_3 = resnet_block(32, kernel_size=ks)
        self.upconv1_2 = resnet_block(32, kernel_size=ks)
        self.upconv1_1 = resnet_block(32, kernel_size=ks)

        self.img_prd = conv(32, 3, kernel_size=ks, stride=1)

    def forward(self, imgs, disps_bi, disp_feature):
        img_left  = imgs[:,:3]
        img_right = imgs[:,3:]

        disp_left  = disps_bi[:, 0]
        disp_right = disps_bi[:, 1]

        # encoder-left
        conv1_left = self.conv1_4(self.conv1_3(self.conv1_2(self.conv1_1(img_left))))
        conv2_left = self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(conv1_left))))
        conv3_left = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(conv2_left))))
        convd_left = self.convd_3(self.convd_2(self.convd_1(conv3_left)))

        # encoder-right
        conv1_right = self.conv1_4(self.conv1_3(self.conv1_2(self.conv1_1(img_right))))
        conv2_right = self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(conv1_right))))
        conv3_right = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(conv2_right))))
        convd_right = self.convd_3(self.convd_2(self.convd_1(conv3_right)))

        b, c, h, w = convd_left.shape

        warp_img_left = disp_warp(img_right, -disp_left*cfg.DATA.DIV_DISP, cuda=True)
        warp_img_right = disp_warp(img_left, disp_right*cfg.DATA.DIV_DISP, cuda=True)
        diff_left = torch.sum(torch.abs(img_left - warp_img_left), 1).view(b,1,*warp_img_left.shape[-2:])
        diff_right = torch.sum(torch.abs(img_right - warp_img_right), 1).view(b,1,*warp_img_right.shape[-2:])
        diff_2_left = nn.functional.adaptive_avg_pool2d(diff_left, (h, w))
        diff_2_right = nn.functional.adaptive_avg_pool2d(diff_right, (h, w))

        disp_2_left = nn.functional.adaptive_avg_pool2d(disp_left, (h, w))
        disp_2_right = nn.functional.adaptive_avg_pool2d(disp_right, (h, w))

        disp_feature_2 = nn.functional.adaptive_avg_pool2d(disp_feature, (h, w))

        depth_aware_left = self.depth_sense_l(torch.cat([disp_feature_2, disp_2_left.view(b,1,h,w)], 1))
        depth_aware_right = self.depth_sense_r(torch.cat([disp_feature_2, disp_2_right.view(b,1,h,w)], 1))

        # the larger, the more accurate
        gate_left  = self.gatenet(diff_2_left)
        gate_right = self.gatenet(diff_2_right)

        warp_convd_left  = disp_warp(convd_right, -disp_2_left)
        warp_convd_right = disp_warp(convd_left, disp_2_right)

        # aggregate features
        agg_left  = convd_left * (1.0-gate_left) + warp_convd_left * gate_left.repeat(1,c,1,1)
        agg_right = convd_right * (1.0-gate_right) + warp_convd_right * gate_right.repeat(1,c,1,1)

        # decoder-left
        cat3_left = self.upconv3_i(torch.cat([convd_left, agg_left, depth_aware_left], 1))
        upconv3_left = self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3_left)))                       # upconv3 feature

        upconv2_u_left = self.upconv2_u(upconv3_left)
        cat2_left = self.upconv2_i(torch.cat([conv2_left, upconv2_u_left],1))
        upconv2_left = self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2_left)))                       # upconv2 feature
        upconv1_u_left = self.upconv1_u(upconv2_left)
        cat1_left = self.upconv1_i(torch.cat([conv1_left, upconv1_u_left], 1))

        upconv1_left = self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1_left)))                       # upconv1 feature
        img_prd_left = self.img_prd(upconv1_left) + img_left                                           # predict img

        # decoder-right
        cat3_right = self.upconv3_i(torch.cat([convd_right, agg_right, depth_aware_right], 1))
        upconv3_right = self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3_right)))                     # upconv3 feature

        upconv2_u_right = self.upconv2_u(upconv3_right)
        cat2_right = self.upconv2_i(torch.cat([conv2_right, upconv2_u_right], 1))
        upconv2_right = self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2_right)))                     # upconv2 feature
        upconv1_u_right = self.upconv1_u(upconv2_right)
        cat1_right = self.upconv1_i(torch.cat([conv1_right, upconv1_u_right], 1))

        upconv1_right = self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1_right)))                     # upconv1 feature
        img_prd_right = self.img_prd(upconv1_right) + img_right                                        # predict img

        imgs_prd = [img_prd_left, img_prd_right]

        diff = [diff_left, diff_right]
        gate = [gate_left, gate_right]

        return imgs_prd, diff, gate
