#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multiscaleloss import *
import torchvision

from time import time

def test_dispnet(cfg, epoch_idx, test_data_loader, dispnet, test_writer):

    # Testing loop
    n_batches = len(test_data_loader)
    test_epe  = dict()
    # Batch average meterics
    batch_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    disp_EPEs = utils.network_utils.AverageMeter()
    test_time = utils.network_utils.AverageMeter()

    batch_end_time = time()
    for batch_idx, (_, images, disps, occs) in enumerate(test_data_loader):
        data_time.update(time() - batch_end_time)

        # Switch models to testing mode
        dispnet.eval();

        with torch.no_grad():
            # Get data from data loader
            disparities = disps
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            imgs = torch.cat(imgs, 1)
            ground_truth_disps = [utils.network_utils.var_or_cuda(disp) for disp in disparities]
            ground_truth_disps = torch.cat(ground_truth_disps, 1)
            occs = [utils.network_utils.var_or_cuda(occ) for occ in occs]
            occs = torch.cat(occs, 1)

            # Test the decoder
            torch.cuda.synchronize()
            test_time_start = time()
            output_disps = dispnet(imgs)
            torch.cuda.synchronize()
            test_time.update(time() - test_time_start)
            print('[TIME] {0}'.format(test_time))

            disp_EPE = cfg.DATA.DIV_DISP * realEPE(output_disps, ground_truth_disps, occs)

            # Append loss and accuracy to average metrics
            disp_EPEs.update(disp_EPE.item(), cfg.CONST.TEST_BATCH_SIZE)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # Print result
            if (batch_idx+1) % cfg.TEST.PRINT_FREQ == 0:
                print('[TEST] [Epoch {0}/{1}][Batch {2}/{3}]\t BatchTime {4}\t DataTime {5}\t\t DispEPE {6}'
                      .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time, disp_EPEs))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM:

                if epoch_idx == 0 or cfg.NETWORK.PHASE in ['test', 'resume']:
                    test_writer.add_image('DispNet/IMG_LEFT'+str(batch_idx+1),
                                          images[0][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)
                    test_writer.add_image('DispNet/IMG_RIGHT'+str(batch_idx+1),
                                          images[1][0][[2,1,0],:,:] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1), epoch_idx+1)

                    gt_disp_left, gt_disp_right = utils.network_utils.graybi2rgb(ground_truth_disps[0])
                    test_writer.add_image('DispNet/DISP_GT_LEFT' +str(batch_idx + 1), gt_disp_left, epoch_idx + 1)
                    test_writer.add_image('DispNet/DISP_GT_RIGHT'+str(batch_idx+1), gt_disp_right, epoch_idx+1)

                b, _, h, w = imgs.size()
                output_disps_up = torch.nn.functional.interpolate(output_disps, size=(h, w), mode = 'bilinear', align_corners=True)
                output_disp_up_left, output_disp_up_right = utils.network_utils.graybi2rgb(output_disps_up[0])
                test_writer.add_image('DispNet/DISP_OUT_LEFT_'+str(batch_idx+1), output_disp_up_left, epoch_idx+1)
                test_writer.add_image('DispNet/DISP_OUT_RIGHT_'+str(batch_idx+1), output_disp_up_right, epoch_idx+1)



    # Output testing results
    print('============================ TEST RESULTS ============================')
    print('[TEST] [Epoch{0}]\t BatchTime_avg {1}\t DataTime_avg {2}\t DispEPE_avg {3}\n'
          .format(cfg.TRAIN.NUM_EPOCHES, batch_time.avg, data_time.avg, disp_EPEs.avg))

    # Add testing results to TensorBoard
    test_writer.add_scalar('DispNet/EpochEPE_1_TEST', disp_EPEs.avg, epoch_idx+1)
    return disp_EPEs.avg
