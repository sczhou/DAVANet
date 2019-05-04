#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import torchvision

from losses.multiscaleloss import *
from time import time

from core.test_disp import test_dispnet


def train_dispnet(cfg, init_epoch, train_data_loader, val_data_loader, dispnet, dispnet_solver,
              dispnet_lr_scheduler, ckpt_dir, train_writer, val_writer, Best_Disp_EPE, Best_Epoch):
    # Training loop
    Best_Disp_EPE    = float('Inf')
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        disp_losses = utils.network_utils.AverageMeter()
        disp_EPEs = utils.network_utils.AverageMeter()
        disp_EPEs_blur = utils.network_utils.AverageMeter()
        disp_EPEs_clear = utils.network_utils.AverageMeter()

        # Adjust learning rate
        dispnet_lr_scheduler.step()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        for batch_idx, (_, images, disps, occs) in enumerate(train_data_loader):
            # Measure data time

            data_time.update(time() - batch_end_time)
            # Get data from data loader
            disparities = disps
            imgs_blur = [utils.network_utils.var_or_cuda(img) for img in images[:2]]
            imgs_clear = [utils.network_utils.var_or_cuda(img) for img in images[2:]]

            imgs_blur  = torch.cat(imgs_blur, 1)
            imgs_clear = torch.cat(imgs_clear, 1)
            ground_truth_disps = [utils.network_utils.var_or_cuda(disp) for disp in disparities]
            ground_truth_disps = torch.cat(ground_truth_disps, 1)
            occs = [utils.network_utils.var_or_cuda(occ) for occ in occs]
            occs = torch.cat(occs, 1)

            # switch models to training mode
            dispnet.train()

            # Train the model
            output_disps_blur = dispnet(imgs_blur)

            disp_loss_blur = multiscaleLoss(output_disps_blur, ground_truth_disps, imgs_blur, occs, cfg.LOSS.MULTISCALE_WEIGHTS)
            disp_EPE_blur = cfg.DATA.DIV_DISP * realEPE(output_disps_blur[0], ground_truth_disps, occs)
            disp_EPEs_blur.update(disp_EPE_blur.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            output_disps_clear = dispnet(imgs_clear)

            disp_loss_clear = multiscaleLoss(output_disps_clear, ground_truth_disps, imgs_clear, occs, cfg.LOSS.MULTISCALE_WEIGHTS)
            disp_EPE_clear = cfg.DATA.DIV_DISP * realEPE(output_disps_clear[0], ground_truth_disps, occs)
            disp_EPEs_clear.update(disp_EPE_clear.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            # Gradient decent
            dispnet_solver.zero_grad()
            disp_loss_clear.backward()
            dispnet_solver.step()

            disp_loss = (disp_loss_blur + disp_loss_clear) / 2.0
            disp_EPE = (disp_EPE_blur + disp_EPE_clear) / 2.0
            disp_losses.update(disp_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            disp_EPEs.update(disp_EPE.item(), cfg.CONST.TRAIN_BATCH_SIZE)


            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('DispNet/BatchLoss_0_TRAIN', disp_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                print(
                    '[TRAIN] [Epoch {0}/{1}][Batch {2}/{3}]\t BatchTime {4}\t DataTime {5}\t DispLoss {6}\t blurEPE {7}\t clearEPE {8}'
                    .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time,
                            disp_losses, disp_EPEs_blur, disp_EPEs_clear))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM:
                img_left_blur = images[0][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_right_blur = images[1][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_left_clear = images[2][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_right_clear = images[3][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                gt_disp_left, gt_disp_right = utils.network_utils.graybi2rgb(ground_truth_disps[0])
                b, _, h, w = imgs_clear.size()
                output_disps_up_blur  = torch.nn.functional.interpolate(output_disps_blur[0], size=(h, w), mode='bilinear', align_corners=True)
                output_disps_up_clear = torch.nn.functional.interpolate(output_disps_clear[0], size=(h, w), mode='bilinear', align_corners=True)
                output_disp_up_left_blur, output_disp_up_right_blur = utils.network_utils.graybi2rgb(output_disps_up_blur[0])
                output_disp_up_left_clear, output_disp_up_right_clear = utils.network_utils.graybi2rgb(output_disps_up_clear[0])
                result = torch.cat([torch.cat([img_left_blur, img_right_blur], 2),
                                    torch.cat([img_left_clear, img_right_clear], 2),
                                    torch.cat([gt_disp_left, gt_disp_right], 2),
                                    torch.cat([output_disp_up_left_blur, output_disp_up_right_blur], 2),
                                    torch.cat([output_disp_up_left_clear, output_disp_up_right_clear], 2)],1)
                result = torchvision.utils.make_grid(result, nrow=1, normalize=True)
                train_writer.add_image('DispNet/TRAIN_RESULT' + str(batch_idx + 1), result, epoch_idx + 1)

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('DispNet/EpochEPE_0_TRAIN', disp_EPEs.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t DispLoss_avg {3}\t DispEPE_avg {4}'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, disp_losses.avg,
                      disp_EPEs.avg))


        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_disp_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                                                 epoch_idx + 1, dispnet, dispnet_solver, Best_Disp_EPE,
                                                 Best_Epoch)
        if disp_EPEs.avg < Best_Disp_EPE:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Disp_EPE = disp_EPEs.avg
            Best_Epoch = epoch_idx + 1
            utils.network_utils.save_disp_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                                                 epoch_idx + 1, dispnet, dispnet_solver, Best_Disp_EPE,
                                                 Best_Epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()

