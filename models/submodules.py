#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import torch.nn as nn
import torch
import numpy as np
from config import cfg

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True)
    )

def predict_disp(in_channels):
    return nn.Conv2d(in_channels,1,kernel_size=3,stride=1,padding=1,bias=True)

def predict_disp_bi(in_channels):
    return nn.Conv2d(in_channels,2,kernel_size=3,stride=1,padding=1,bias=True)

def up_disp_bi():
    return nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

def predict_occ(in_channels):
    return nn.Conv2d(in_channels,1,kernel_size=3,stride=1,padding=1,bias=True)

def predict_occ_bi(in_channels):
    return nn.Conv2d(in_channels,2,kernel_size=3,stride=1,padding=1,bias=True)

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


def gatenet(bias=True):
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
        resnet_block(16, kernel_size=1),
        nn.Conv2d(16, 1, kernel_size=1, padding=0),
        nn.Sigmoid()
    )

def depth_sense(in_channels, out_channels, kernel_size=3, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,padding=((kernel_size - 1) // 2)*dilation, bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True),
        resnet_block(out_channels, kernel_size= 3),
    )

def conv2x(in_channels, kernel_size=3,dilation=[1,1], bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE,inplace=True),
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE, inplace=True)
    )


def ms_dilate_block(in_channels, kernel_size=3, dilation=[1,1,1,1], bias=True):
    return MSDilateBlock(in_channels, kernel_size, dilation, bias)

class MSDilateBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  nn.Conv2d(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out


def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output
