#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import sys
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import models
from models.DispNet_Bi import DispNet_Bi
from models.DeblurNet import DeblurNet
from models.StereoDeblurNet import StereoDeblurNet

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from core.train_disp import train_dispnet
from core.test_disp import test_dispnet
from core.train_deblur import train_deblurnet
from core.test_deblur import test_deblurnet
from core.train_stereodeblur import train_stereodeblurnet
from core.test_stereodeblur import test_stereodeblurnet
from losses.multiscaleloss import *

def bulid_net(cfg):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD, div_disp=cfg.DATA.DIV_DISP),
        utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
        utils.data_transforms.RandomVerticalFlip(),
        utils.data_transforms.RandomColorChannel(),
        utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        utils.data_transforms.ToTensor(),
    ])

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD, div_disp=cfg.DATA.DIV_DISP),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME]()
    if cfg.NETWORK.PHASE in ['train', 'resume']:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)

    test_data_loader   = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    # Set up networks
    dispnet = models.__dict__[cfg.NETWORK.DISPNETARCH].__dict__[cfg.NETWORK.DISPNETARCH]()
    deblurnet = models.__dict__[cfg.NETWORK.DEBLURNETARCH].__dict__[cfg.NETWORK.DEBLURNETARCH]()


    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.DISPNETARCH,
                                                utils.network_utils.count_parameters(dispnet)))

    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.DEBLURNETARCH,
                                                utils.network_utils.count_parameters(deblurnet)))

    # Initialize weights of networks
    dispnet.apply(utils.network_utils.init_weights_kaiming)
    deblurnet.apply(utils.network_utils.init_weights_xavier)
    # Set up solver
    dispnet_solver   = torch.optim.Adam(filter(lambda p: p.requires_grad, dispnet.parameters()), lr=cfg.TRAIN.DISPNET_LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
    deblurnet_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, deblurnet.parameters()), lr=cfg.TRAIN.DEBLURNET_LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    if torch.cuda.is_available():
        dispnet = torch.nn.DataParallel(dispnet).cuda()
        deblurnet = torch.nn.DataParallel(deblurnet).cuda()

    # Load pretrained model if exists
    init_epoch       = 0
    Best_Epoch       = -1
    Best_Disp_EPE    = float('Inf')
    Best_Img_PSNR    = 0
    if cfg.NETWORK.PHASE in ['test', 'resume']:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)

        if cfg.NETWORK.MODULE == 'dispnet':
            dispnet.load_state_dict(checkpoint['dispnet_state_dict'])
            init_epoch = checkpoint['epoch_idx']+1
            Best_Disp_EPE = checkpoint['Best_Disp_EPE']
            Best_Epoch = checkpoint['Best_Epoch']
            dispnet_solver.load_state_dict(checkpoint['dispnet_solver_state_dict'])
            print('[INFO] {0} Recover complete. Current epoch #{1}, Best_Disp_EPE = {2} at epoch #{3}.' \
                  .format(dt.now(), init_epoch, Best_Disp_EPE, Best_Epoch))
        elif cfg.NETWORK.MODULE == 'deblurnet':
            deblurnet.load_state_dict(checkpoint['deblurnet_state_dict'])
            init_epoch = checkpoint['epoch_idx']+1
            Best_Img_PSNR = checkpoint['Best_Img_PSNR']
            Best_Epoch = checkpoint['Best_Epoch']
            deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])
            print('[INFO] {0} Recover complete. Current epoch #{1}, Best_Img_PSNR = {2} at epoch #{3}.' \
                  .format(dt.now(), init_epoch, Best_Img_PSNR, Best_Epoch))
            init_epoch = 0
        elif cfg.NETWORK.MODULE == 'all':
            Best_Img_PSNR = checkpoint['Best_Img_PSNR']
            dispnet.load_state_dict(checkpoint['dispnet_state_dict'])
            deblurnet.load_state_dict(checkpoint['deblurnet_state_dict'])
            print('[INFO] {0} Recover complete. Best_Img_PSNR = {1}'.format(dt.now(), Best_Img_PSNR))


    # Set up learning rate scheduler to decay learning rates dynamically
    dispnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(dispnet_solver,
                                                                   milestones=cfg.TRAIN.DISPNET_LR_MILESTONES,
                                                                   gamma=cfg.TRAIN.LEARNING_RATE_DECAY)
    deblurnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(deblurnet_solver,
                                                                   milestones=cfg.TRAIN.DEBLURNET_LR_MILESTONES,
                                                                   gamma=cfg.TRAIN.LEARNING_RATE_DECAY)

    # Summary writer for TensorBoard
    if cfg.NETWORK.MODULE == 'dispnet':
        output_dir   = os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat()+'_'+cfg.NETWORK.DISPNETARCH, '%s')
    elif cfg.NETWORK.MODULE == 'deblurnet':
        output_dir   = os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat()+'_'+cfg.NETWORK.DEBLURNETARCH, '%s')
    elif cfg.NETWORK.MODULE == 'all':
        output_dir = os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat() + '_' + cfg.NETWORK.DEBLURNETARCH, '%s')
    log_dir      = output_dir % 'logs'
    ckpt_dir     = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer  = SummaryWriter(os.path.join(log_dir, 'test'))


    if cfg.NETWORK.PHASE in ['train', 'resume']:
        # train and val
        if cfg.NETWORK.MODULE == 'dispnet':
            train_dispnet(cfg, init_epoch, train_data_loader, test_data_loader, dispnet, dispnet_solver,
                      dispnet_lr_scheduler, ckpt_dir, train_writer, test_writer, Best_Disp_EPE, Best_Epoch)
            return
        elif cfg.NETWORK.MODULE == 'deblurnet':
            train_deblurnet(cfg, init_epoch, train_data_loader, test_data_loader, deblurnet, deblurnet_solver,
                              deblurnet_lr_scheduler, ckpt_dir, train_writer, test_writer, Best_Img_PSNR, Best_Epoch)
            return
        elif cfg.NETWORK.MODULE == 'all':
            train_stereodeblurnet(cfg, init_epoch, train_data_loader, test_data_loader,
                                  dispnet, dispnet_solver, dispnet_lr_scheduler,
                                  deblurnet, deblurnet_solver, deblurnet_lr_scheduler,
                                  ckpt_dir, train_writer, test_writer,
                                  Best_Disp_EPE, Best_Img_PSNR, Best_Epoch)

    else:
        assert os.path.exists(cfg.CONST.WEIGHTS),'[FATAL] Please specify the file path of checkpoint!'
        if cfg.NETWORK.MODULE == 'dispnet':
            test_dispnet(cfg, init_epoch, test_data_loader, dispnet, test_writer)
            return
        elif cfg.NETWORK.MODULE == 'deblurnet':
            test_deblurnet(cfg, init_epoch, test_data_loader, deblurnet, test_writer)
            return
        elif cfg.NETWORK.MODULE == 'all':
            test_stereodeblurnet(cfg, init_epoch, test_data_loader, dispnet, deblurnet, test_writer)
