#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
import os
import sys
import torch.backends.cudnn
import torch.utils.data
import numpy as np
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multiscaleloss import *
from time import time
import cv2

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test_deblurnet(cfg, epoch_idx, test_data_loader, deblurnet, test_writer):

    # Testing loop
    n_batches = len(test_data_loader)
    test_epe  = dict()
    # Batch average meterics
    batch_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    img_PSNRs = utils.network_utils.AverageMeter()
    batch_end_time = time()

    test_psnr = dict()
    g_names= 'init'
    save_num = 0
    for batch_idx, (names, images, DISPs, OCCs) in enumerate(test_data_loader):
        data_time.update(time() - batch_end_time)
        if not g_names == names:
            g_names = names
            save_num = 0
        save_num = save_num+1
        # Switch models to testing mode
        deblurnet.eval()

        if cfg.NETWORK.PHASE == 'test':
            assert (len(names) == 1)
            name = names[0]
            if not name in test_psnr:
                test_psnr[name] = {
                    'n_samples': 0,
                    'psnr': []
                }

        with torch.no_grad():
            # Get data from data loader
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            img_blur_left, img_blur_right, img_clear_left, img_clear_right = imgs

            # Test the decoder
            output_img_clear_left = deblurnet(img_blur_left)
            output_img_clear_right = deblurnet(img_blur_right)

            # Append loss and accuracy to average metrics
            img_PSNR = PSNR(output_img_clear_left, img_clear_left) / 2 + PSNR(output_img_clear_right, img_clear_right) / 2
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TEST_BATCH_SIZE)

            if cfg.NETWORK.PHASE == 'test':
                test_psnr[name]['n_samples'] += 1
                test_psnr[name]['psnr'].append(img_PSNR)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # Print result
            if (batch_idx+1) % cfg.TEST.PRINT_FREQ == 0:
                print('[TEST] [Epoch {0}/{1}][Batch {2}/{3}]\t BatchTime {4}\t DataTime {5}\t\t ImgPSNR {6}'
                      .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time, img_PSNRs))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM:
                if epoch_idx == 0 or cfg.NETWORK.PHASE in ['test', 'resume']:
                    test_writer.add_image('DeblurNet/IMG_BLUR_LEFT'+str(batch_idx+1),
                                          images[0][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)
                    test_writer.add_image('DeblurNet/IMG_BLUR_RIGHT'+str(batch_idx+1),
                                          images[1][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)
                    test_writer.add_image('DeblurNet/IMG_CLEAR_LEFT' + str(batch_idx + 1),
                                          images[2][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx + 1)
                    test_writer.add_image('DeblurNet/IMG_CLEAR_RIGHT' + str(batch_idx + 1),
                                          images[3][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx + 1)

                test_writer.add_image('DeblurNet/OUT_IMG_CLEAR_LEFT'+str(batch_idx+1), output_img_clear_left[0][[2,1,0],:,:].cpu().clamp(0.0,1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)
                test_writer.add_image('DeblurNet/OUT_IMG_CLEAR_RIGHT'+str(batch_idx+1), output_img_clear_right[0][[2,1,0],:,:].cpu().clamp(0.0,1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)

            if cfg.NETWORK.PHASE == 'test':
                left_out_dir = os.path.join(cfg.DIR.OUT_PATH,'single',names[0],'left')
                right_out_dir = os.path.join(cfg.DIR.OUT_PATH,'single',names[0],'right')
                if not os.path.isdir(left_out_dir):
                    mkdir(left_out_dir)
                if not os.path.isdir(right_out_dir):
                    mkdir(right_out_dir)
                print(left_out_dir+'/'+str(save_num).zfill(4)+'.png')
                cv2.imwrite(left_out_dir+'/'+str(save_num).zfill(4)+'.png', (output_img_clear_left.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8),
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 5])

                cv2.imwrite(right_out_dir + '/' + str(save_num).zfill(4) + '.png',
                            (output_img_clear_right.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8),[int(cv2.IMWRITE_PNG_COMPRESSION), 5])

    if cfg.NETWORK.PHASE == 'test':

        # Output test results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        for name in test_psnr:
            test_psnr[name]['psnr'] = np.mean(test_psnr[name]['psnr'], axis=0)
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))

        result_file = open(os.path.join(cfg.DIR.OUT_PATH, 'test_result.txt'), 'w')
        sys.stdout = result_file
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        for name in test_psnr:
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))
        result_file.close()
    else:
        # Output val results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        print('[TEST] [Epoch{0}]\t BatchTime_avg {1}\t DataTime_avg {2}\t ImgPSNR_avg {3}\n'
              .format(cfg.TRAIN.NUM_EPOCHES, batch_time.avg, data_time.avg, img_PSNRs.avg))

        # Add testing results to TensorBoard
        test_writer.add_scalar('DeblurNet/EpochPSNR_1_TEST', img_PSNRs.avg, epoch_idx + 1)

        return img_PSNRs.avg