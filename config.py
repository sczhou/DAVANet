#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

from easydict import EasyDict as edict
import socket

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.DEVICE                        = 'all'                   # '0'
__C.CONST.NUM_WORKER                    = 1                       # number of data workers
__C.CONST.WEIGHTS                       = '/data/code/StereodeblurNet-release/ckpt/best-ckpt.pth.tar'
__C.CONST.TRAIN_BATCH_SIZE              = 1
__C.CONST.TEST_BATCH_SIZE               = 1


#
# Dataset
#
__C.DATASET                             = edict()
__C.DATASET.DATASET_NAME                = 'StereoDeblur'          # FlyingThings3D, StereoDeblur
__C.DATASET.WITH_MASK                   = True

if cfg.DATASET.DATASET_NAME == 'StereoDeblur':
    __C.DATASET.SPARSE                  = True
else:
    __C.DATASET.SPARSE                  = False

#
# Directories
#
__C.DIR                                 = edict()
__C.DIR.OUT_PATH = '/data/code/StereodeblurNet/output'

# For FlyingThings3D Dataset
if cfg.DATASET.DATASET_NAME == 'FlyingThings3D':
    __C.DIR.DATASET_JSON_FILE_PATH          = './datasets/flyingthings3d.json'
    __C.DIR.DATASET_ROOT                    = '/data/scene_flow/FlyingThings3D/'
    __C.DIR.IMAGE_LEFT_PATH                 = __C.DIR.DATASET_ROOT + '%s/%s/%s/%s/left/%s.png'
    __C.DIR.IMAGE_RIGHT_PATH                = __C.DIR.DATASET_ROOT + '%s/%s/%s/%s/right/%s.png'
    __C.DIR.DISPARITY_LEFT_PATH             = __C.DIR.DATASET_ROOT + 'disparity/%s/%s/%s/left/%s.pfm'
    __C.DIR.DISPARITY_RIGHT_PATH            = __C.DIR.DATASET_ROOT + 'disparity/%s/%s/%s/right/%s.pfm'

# For Stereo_Blur_Dataset
elif cfg.DATASET.DATASET_NAME == 'StereoDeblur':
      __C.DIR.DATASET_JSON_FILE_PATH        = './datasets/stereo_deblur_data.json'
      __C.DIR.DATASET_ROOT                  = '/data1/stereo_deblur_data_final_gamma/'
      __C.DIR.IMAGE_LEFT_BLUR_PATH          = __C.DIR.DATASET_ROOT + '%s/image_left_blur_ga/%s.png'
      __C.DIR.IMAGE_LEFT_CLEAR_PATH         = __C.DIR.DATASET_ROOT + '%s/image_left/%s.png'
      __C.DIR.IMAGE_RIGHT_BLUR_PATH         = __C.DIR.DATASET_ROOT + '%s/image_right_blur_ga/%s.png'
      __C.DIR.IMAGE_RIGHT_CLEAR_PATH        = __C.DIR.DATASET_ROOT + '%s/image_right/%s.png'
      __C.DIR.DISPARITY_LEFT_PATH           = __C.DIR.DATASET_ROOT + '%s/disparity_left/%s.exr'
      __C.DIR.DISPARITY_RIGHT_PATH          = __C.DIR.DATASET_ROOT + '%s/disparity_right/%s.exr'

#
# data augmentation
#
__C.DATA                                = edict()
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.DIV_DISP                       = 40.0                    # 40.0 for disparity
__C.DATA.CROP_IMG_SIZE                  = [256, 256]              # Crop image size: height, width
__C.DATA.GAUSSIAN                       = [0, 1e-4]               # mu, std_var
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue

#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.DISPNETARCH                 = 'DispNet_Bi'            # available options: DispNet_Bi
__C.NETWORK.DEBLURNETARCH               = 'StereoDeblurNet'       # available options: DeblurNet, StereoDeblurNet
__C.NETWORK.LEAKY_VALUE                 = 0.1
__C.NETWORK.BATCHNORM                   = False
__C.NETWORK.PHASE                       = 'train'                 # available options: 'train', 'test', 'resume'
__C.NETWORK.MODULE                      = 'all'                   # available options: 'dispnet', 'deblurnet', 'all'
#
# Training
#

__C.TRAIN                               = edict()
__C.TRAIN.USE_PERCET_LOSS               = True
__C.TRAIN.NUM_EPOCHES                   = 400                     # maximum number of epoches
__C.TRAIN.BRIGHTNESS                    = .25
__C.TRAIN.CONTRAST                      = .25
__C.TRAIN.SATURATION                    = .25
__C.TRAIN.HUE                           = .25
__C.TRAIN.DISPNET_LEARNING_RATE         = 1e-6
__C.TRAIN.DEBLURNET_LEARNING_RATE       = 1e-4
__C.TRAIN.DISPNET_LR_MILESTONES         = [100,200,300]
__C.TRAIN.DEBLURNET_LR_MILESTONES       = [80,160,240]
__C.TRAIN.LEARNING_RATE_DECAY           = 0.1                     # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                     # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                     # regularization of weight, default: 0
__C.TRAIN.PRINT_FREQ                    = 10
__C.TRAIN.SAVE_FREQ                     = 5                       # weights will be overwritten every save_freq epoch

__C.LOSS                                = edict()
__C.LOSS.MULTISCALE_WEIGHTS             = [0.3, 0.3, 0.2, 0.1, 0.1]

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.VISUALIZATION_NUM              = 3
__C.TEST.PRINT_FREQ                     = 5
if __C.NETWORK.PHASE == 'test':
    __C.CONST.TEST_BATCH_SIZE           = 1
