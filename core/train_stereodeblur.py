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

from core.test_stereodeblur import test_stereodeblurnet
from models.VGG19 import VGG19


def train_stereodeblurnet(cfg, init_epoch, train_data_loader, val_data_loader,
                                  dispnet, dispnet_solver, dispnet_lr_scheduler,
                                  deblurnet, deblurnet_solver, deblurnet_lr_scheduler,
                                  ckpt_dir, train_writer, val_writer,
                                  Disp_EPE, Best_Img_PSNR, Best_Epoch):
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        disp_EPEs = utils.network_utils.AverageMeter()
        deblur_mse_losses = utils.network_utils.AverageMeter()
        if cfg.TRAIN.USE_PERCET_LOSS == True:
            deblur_percept_losses = utils.network_utils.AverageMeter()
        deblur_losses = utils.network_utils.AverageMeter()
        img_PSNRs = utils.network_utils.AverageMeter()

        # Adjust learning rate
        dispnet_lr_scheduler.step()
        deblurnet_lr_scheduler.step()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        vggnet = VGG19()
        if torch.cuda.is_available():
            vggnet = torch.nn.DataParallel(vggnet).cuda()

        for batch_idx, (_, images, disps, occ_masks) in enumerate(train_data_loader):
            # Measure data time

            data_time.update(time() - batch_end_time)
            # Get data from data loader
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            img_blur_left, img_blur_right, img_clear_left, img_clear_right = imgs

            imgs_blur = torch.cat([img_blur_left, img_blur_right], 1)
            ground_truth_disps = [utils.network_utils.var_or_cuda(disp) for disp in disps]
            ground_truth_disps = torch.cat(ground_truth_disps, 1)
            occ_masks = [utils.network_utils.var_or_cuda(occ_mask) for occ_mask in occ_masks]
            occ_masks = torch.cat(occ_masks, 1)

            # switch models to training mode
            dispnet.train()
            deblurnet.train()

            # Train the model
            output_disps = dispnet(imgs_blur)

            output_disp_feature = output_disps[-1]
            output_disps = output_disps[:-1]
            imgs_prd, output_diffs, output_masks = deblurnet(imgs_blur, output_disps[0], output_disp_feature)

            disp_EPE = cfg.DATA.DIV_DISP * realEPE(output_disps[0], ground_truth_disps, occ_masks)
            disp_EPEs.update(disp_EPE.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            # deblur loss
            deblur_mse_left_loss = mseLoss(imgs_prd[0], img_clear_left)
            deblur_mse_right_loss = mseLoss(imgs_prd[1], img_clear_right)
            deblur_mse_loss = (deblur_mse_left_loss + deblur_mse_right_loss) / 2
            deblur_mse_losses.update(deblur_mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            if cfg.TRAIN.USE_PERCET_LOSS == True:
                deblur_percept_left_loss = perceptualLoss(imgs_prd[0], img_clear_left, vggnet)
                deblur_percept_right_loss = perceptualLoss(imgs_prd[1], img_clear_right, vggnet)
                deblur_percept_loss = (deblur_percept_left_loss + deblur_percept_right_loss) / 2
                deblur_percept_losses.update(deblur_percept_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
                deblur_loss = deblur_mse_loss + 0.01 * deblur_percept_loss
            else:
                deblur_loss = deblur_mse_loss
            deblur_losses.update(deblur_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            img_PSNR = (PSNR(imgs_prd[0], img_clear_left) + PSNR(imgs_prd[1], img_clear_right)) / 2
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            deblurnet_solver.zero_grad()
            deblurnet_loss = deblur_loss
            deblurnet_loss.backward()
            deblurnet_solver.step()

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx

            train_writer.add_scalar('StereoDeblurNet/DeblurLoss_0_TRAIN', deblur_loss.item(), n_itr)
            train_writer.add_scalar('StereoDeblurNet/DeblurMSELoss_0_TRAIN', deblur_mse_loss.item(), n_itr)
            if cfg.TRAIN.USE_PERCET_LOSS == True:
                train_writer.add_scalar('StereoDeblurNet/DeblurPerceptLoss_0_TRAIN', deblur_percept_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                print(
                    '[TRAIN] [Ech {0}/{1}][Bch {2}/{3}] BT {4} DT {5} EPE {6} DeblurLoss {7} [{8}, {9}] PSNR {10}'
                        .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time,
                                 disp_EPEs, deblur_losses, deblur_mse_losses, deblur_percept_losses, img_PSNRs))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM:
                img_blur_left = images[0][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_blur_right = images[1][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_clear_left = images[2][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_clear_right = images[3][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_out_left = imgs_prd[0][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_out_right = imgs_prd[1][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                disp_gt_left, disp_gt_right = utils.network_utils.graybi2rgb(ground_truth_disps[0])
                b, _, h, w = imgs[0].shape
                diff_out_left, diff_out_right = utils.network_utils.graybi2rgb(torch.cat(output_diffs, 1)[0])
                output_masks = torch.nn.functional.interpolate(torch.cat(output_masks, 1), size=(h, w), mode='bilinear', align_corners=True)
                mask_out_left, mask_out_right = utils.network_utils.graybi2rgb(output_masks[0])
                disp_out_left, disp_out_right = utils.network_utils.graybi2rgb(output_disps[0][0])
                result = torch.cat([torch.cat([img_blur_left, img_blur_right], 2),
                                    torch.cat([img_clear_left, img_clear_right], 2),
                                    torch.cat([img_out_left, img_out_right], 2),
                                    torch.cat([disp_gt_left, disp_gt_right], 2),
                                    torch.cat([disp_out_left, disp_out_right], 2),
                                    torch.cat([diff_out_left, diff_out_right], 2),
                                    torch.cat([mask_out_left, mask_out_right], 2)], 1)
                result = torchvision.utils.make_grid(result, nrow=1, normalize=True)
                train_writer.add_image('StereoDeblurNet/TRAIN_RESULT' + str(batch_idx + 1), result, epoch_idx + 1)

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('StereoDeblurNet/EpochEPE_0_TRAIN', disp_EPEs.avg, epoch_idx + 1)
        train_writer.add_scalar('StereoDeblurNet/EpochPSNR_0_TRAIN', img_PSNRs.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t DispEPE_avg {3}\t ImgPSNR_avg {4}'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, disp_EPEs.avg, img_PSNRs.avg))

        # Validate the training models
        Disp_EPE, img_PSNR = test_stereodeblurnet(cfg, epoch_idx, val_data_loader, dispnet, deblurnet, val_writer)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                                                      epoch_idx + 1, dispnet, dispnet_solver, deblurnet, deblurnet_solver, \
                                                      Disp_EPE, Best_Img_PSNR, Best_Epoch)
        if img_PSNR >= Best_Img_PSNR:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Img_PSNR = img_PSNR
            Best_Epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                                                      epoch_idx + 1, dispnet, dispnet_solver, deblurnet, deblurnet_solver, \
                                                      Disp_EPE, Best_Img_PSNR, Best_Epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()

