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

from core.test_deblur import test_deblurnet
from models.VGG19 import VGG19


def train_deblurnet(cfg, init_epoch, train_data_loader, val_data_loader, deblurnet, deblurnet_solver,
                    deblurnet_lr_scheduler, ckpt_dir, train_writer, val_writer, Best_Img_PSNR, Best_Epoch):
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        test_time = utils.network_utils.AverageMeter()
        deblur_losses = utils.network_utils.AverageMeter()
        mse_losses = utils.network_utils.AverageMeter()
        if cfg.TRAIN.USE_PERCET_LOSS:
            percept_losses = utils.network_utils.AverageMeter()
        img_PSNRs = utils.network_utils.AverageMeter()

        # Adjust learning rate
        deblurnet_lr_scheduler.step()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        if cfg.TRAIN.USE_PERCET_LOSS:
            vggnet = VGG19()
            if torch.cuda.is_available():
                vggnet = torch.nn.DataParallel(vggnet).cuda()

        for batch_idx, (_, images, DISPs, OCCs) in enumerate(train_data_loader):
            # Measure data time

            data_time.update(time() - batch_end_time)
            # Get data from data loader
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            img_blur_left, img_blur_right, img_clear_left, img_clear_right = imgs

            # switch models to training mode
            deblurnet.train()

            output_img_clear_left = deblurnet(img_blur_left)

            mse_left_loss  = mseLoss(output_img_clear_left, img_clear_left)
            if cfg.TRAIN.USE_PERCET_LOSS:
                percept_left_loss  = perceptualLoss(output_img_clear_left, img_clear_left, vggnet)
                deblur_left_loss  = mse_left_loss + 0.01 * percept_left_loss
            else:
                deblur_left_loss = mse_left_loss

            img_PSNR_left = PSNR(output_img_clear_left, img_clear_left)

            # Gradient decent
            deblurnet_solver.zero_grad()
            deblur_left_loss.backward()

            # For right
            output_img_clear_right = deblurnet(img_blur_right)
            mse_right_loss = mseLoss(output_img_clear_right, img_clear_right)
            if cfg.TRAIN.USE_PERCET_LOSS:
                percept_right_loss = perceptualLoss(output_img_clear_right, img_clear_right, vggnet)
                deblur_right_loss = mse_right_loss + 0.01 * percept_right_loss
            else:
                deblur_right_loss = mse_right_loss

            img_PSNR_right = PSNR(output_img_clear_right, img_clear_right)

            # Gradient decent
            deblurnet_solver.zero_grad()
            deblur_right_loss.backward()
            deblurnet_solver.step()

            mse_loss = (mse_left_loss + mse_right_loss) / 2
            mse_losses.update(mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            if cfg.TRAIN.USE_PERCET_LOSS:
                percept_loss = 0.01 *(percept_left_loss + percept_right_loss) / 2
                percept_losses.update(percept_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            deblur_loss = (deblur_left_loss + deblur_right_loss) / 2
            deblur_losses.update(deblur_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            img_PSNR = img_PSNR_left / 2 + img_PSNR_right / 2
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('DeblurNet/MSELoss_0_TRAIN', mse_loss.item(), n_itr)
            if cfg.TRAIN.USE_PERCET_LOSS:
                train_writer.add_scalar('DeblurNet/PerceptLoss_0_TRAIN', percept_loss.item(), n_itr)
            train_writer.add_scalar('DeblurNet/DeblurLoss_0_TRAIN', deblur_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                if cfg.TRAIN.USE_PERCET_LOSS:
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t BT {4}\t DT {5}\t  Loss {6} [{7}, {8}]\t PSNR {9}'
                        .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time,
                                deblur_losses, mse_losses, percept_losses, img_PSNRs))
                else:
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t BT {4}\t DT {5}\t  DeblurLoss {6} \t PSNR {7}'
                          .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time,
                                  deblur_losses, img_PSNRs))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM:

                img_left_blur = images[0][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_right_blur = images[1][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_left_clear = images[2][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_right_clear = images[3][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                out_left = output_img_clear_left[0][[2,1,0],:,:].cpu().clamp(0.0,1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                out_right = output_img_clear_right[0][[2,1,0],:,:].cpu().clamp(0.0,1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                result = torch.cat([torch.cat([img_left_blur, img_right_blur], 2),torch.cat([img_left_clear, img_right_clear], 2),torch.cat([out_left, out_right], 2)],1)
                result = torchvision.utils.make_grid(result, nrow=1, normalize=True)
                train_writer.add_image('DeblurNet/TRAIN_RESULT' + str(batch_idx + 1), result, epoch_idx + 1)


        # Append epoch loss to TensorBoard
        train_writer.add_scalar('DeblurNet/EpochPSNR_0_TRAIN', img_PSNRs.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t DeblurLoss_avg {3}\t ImgPSNR_avg {4}'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, deblur_losses.avg,
                      img_PSNRs.avg))

        # Validate the training models
        img_PSNR = test_deblurnet(cfg, epoch_idx, val_data_loader, deblurnet, val_writer)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_deblur_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                                                 epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR,
                                                 Best_Epoch)
        if img_PSNR > Best_Img_PSNR:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Img_PSNR = img_PSNR
            Best_Epoch = epoch_idx + 1
            utils.network_utils.save_deblur_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                                                 epoch_idx + 1, deblurnet, deblurnet_solver, Best_Img_PSNR,
                                                 Best_Epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


