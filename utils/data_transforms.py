#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''


import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from config import cfg
from PIL import Image
import random
import numbers
class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, disps):
        for t in self.transforms:
            inputs,disps = t(inputs, disps)
        return inputs, disps


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, inputs, disps):
        inputs = [Image.fromarray(np.uint8(inp)) for inp in inputs]
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            inputs = [F.adjust_brightness(inp, brightness_factor) for inp in inputs]

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            inputs = [F.adjust_contrast(inp, contrast_factor) for inp in inputs]

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            inputs = [F.adjust_saturation(inp, saturation_factor) for inp in inputs]

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            inputs = [F.adjust_hue(inp, hue_factor) for inp in inputs]

        inputs = [np.asarray(inp) for inp in inputs]
        inputs = [inp.clip(0,255) for inp in inputs]

        return inputs, disps

class RandomColorChannel(object):
    def __call__(self, inputs, disps):
        random_order = np.random.permutation(3)
        inputs = [inp[:,:,random_order] for inp in inputs]

        return inputs, disps

class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, inputs, disps):

        shape = inputs[0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        # only apply to blurry images
        inputs[0] = inputs[0]+gaussian_noise
        inputs[1] = inputs[1]+gaussian_noise

        inputs = [inp.clip(0, 1) for inp in inputs]

        return inputs, disps

class Normalize(object):
    def __init__(self, mean, std, div_disp):
        self.mean = mean
        self.std  = std
        self.div_disp = div_disp
    def __call__(self, inputs, disps):
        assert(all([isinstance(inp, np.ndarray) for inp in inputs]))
        inputs = [inp/self.std -self.mean for inp in inputs]
        disps = [d/self.div_disp for d in disps]
        return inputs, disps

class CenterCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""

        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, inputs, disps):
        input_size_h, input_size_w, _ = inputs[0].shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))

        inputs = [inp[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for inp in inputs]
        disps  = [disp[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for disp in disps]

        return inputs, disps

class RandomCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, inputs, disps):
        input_size_h, input_size_w, _ = inputs[0].shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)
        inputs = [inp[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for inp in inputs]
        disps = [disp[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for disp in disps]

        return inputs, disps

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, inputs, disps):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            inputs[0] = np.copy(np.fliplr(inputs[1]))
            inputs[1] = np.copy(np.fliplr(inputs[0]))
            inputs[2] = np.copy(np.fliplr(inputs[3]))
            inputs[3] = np.copy(np.fliplr(inputs[2]))

            disps[0]  = np.copy(np.fliplr(disps[1]))
            disps[1]  = np.copy(np.fliplr(disps[0]))

        return inputs, disps


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""
    def __call__(self, inputs, disps):
        if random.random() < 0.5:
            inputs = [np.copy(np.flipud(inp)) for inp in inputs]
            disps  = [np.copy(np.flipud(disp)) for disp in disps]
        return inputs, disps


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, inputs, disps):
        assert(isinstance(inputs[0], np.ndarray) and isinstance(inputs[1], np.ndarray))
        inputs = [np.transpose(inp, (2, 0, 1)) for inp in inputs]
        inputs_tensor = [torch.from_numpy(inp).float() for inp in inputs]

        assert(isinstance(disps[0], np.ndarray) and isinstance(disps[1], np.ndarray))
        disps_tensor = [torch.from_numpy(d) for d in disps]
        disps_tensor = [d.view(1, d.size()[0],d.size()[1]).float() for d in disps_tensor]
        return inputs_tensor, disps_tensor

