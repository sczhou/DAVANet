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

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test_stereodeblurnet(cfg, epoch_idx, test_data_loader, dispnet, deblurnet, test_writer):

    # Testing loop
    n_batches = len(test_data_loader)
    # Batch average meterics
    batch_time = utils.network_utils.AverageMeter()
    test_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    disp_EPEs = utils.network_utils.AverageMeter()
    img_PSNRs = utils.network_utils.AverageMeter()
    batch_end_time = time()
    test_psnr = dict()
    g_names= 'init'
    save_num = 0

    for batch_idx, (names, images, disps, occs) in enumerate(test_data_loader):
        data_time.update(time() - batch_end_time)
        if not g_names == names:
            g_names = names
            save_num = 0
        save_num = save_num+1
        # Switch models to testing mode
        dispnet.eval()
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
            disparities = disps
            imgs = [utils.network_utils.var_or_cuda(img) for img in images]
            img_blur_left, img_blur_right, img_clear_left, img_clear_right = imgs
            imgs_blur = torch.cat([img_blur_left, img_blur_right], 1)

            ground_truth_disps = [utils.network_utils.var_or_cuda(disp) for disp in disparities]
            ground_truth_disps = torch.cat(ground_truth_disps, 1)
            occs = [utils.network_utils.var_or_cuda(occ) for occ in occs]
            occs = torch.cat(occs, 1)

            # Test the dispnet
            # torch.cuda.synchronize()
            # test_time_start = time()

            output_disps = dispnet(imgs_blur)

            output_disp_feature = output_disps[1]
            output_disps = output_disps[0]

            # Test the deblurnet
            imgs_prd, output_diffs, output_masks= deblurnet(imgs_blur, output_disps, output_disp_feature)

            # torch.cuda.synchronize()
            # test_time.update(time() - test_time_start)
            # print('[TIME] {0}'.format(test_time))
            disp_EPE = cfg.DATA.DIV_DISP * realEPE(output_disps, ground_truth_disps, occs)
            disp_EPEs.update(disp_EPE.item(), cfg.CONST.TEST_BATCH_SIZE)

            img_PSNR = (PSNR(imgs_prd[0], img_clear_left) + PSNR(imgs_prd[1], img_clear_right)) / 2
            img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            if cfg.NETWORK.PHASE == 'test':
                test_psnr[name]['n_samples'] += 1
                test_psnr[name]['psnr'].append(img_PSNR)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            # Print result
            if (batch_idx+1) % cfg.TEST.PRINT_FREQ == 0:
                print('[TEST] [Epoch {0}/{1}][Batch {2}/{3}]\t BatchTime {4}\t DataTime {5}\t DispEPE {6}\t imgPSNR {7}'
                      .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time, disp_EPEs, img_PSNRs))

            if batch_idx < cfg.TEST.VISUALIZATION_NUM and cfg.NETWORK.PHASE in ['train', 'resume']:


                if epoch_idx == 0 or cfg.NETWORK.PHASE in ['test', 'resume']:
                    img_blur_left = images[0][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    img_blur_right = images[1][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    img_clear_left = images[2][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    img_clear_right = images[3][0][[2, 1, 0], :, :] + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    test_writer.add_image('StereoDeblurNet/IMG_BLUR' + str(batch_idx + 1),
                                          torch.cat([img_blur_left, img_blur_right], 2), epoch_idx + 1)

                    test_writer.add_image('StereoDeblurNet/IMG_CLEAR' + str(batch_idx + 1),
                                          torch.cat([img_clear_left, img_clear_right], 2), epoch_idx + 1)

                    gt_disp_left, gt_disp_right = utils.network_utils.graybi2rgb(ground_truth_disps[0])
                    test_writer.add_image('StereoDeblurNet/DISP_GT' +str(batch_idx + 1),
                                          torch.cat([gt_disp_left, gt_disp_right], 2), epoch_idx + 1)

                b, _, h, w = imgs[0].size()
                diff_out_left, diff_out_right = utils.network_utils.graybi2rgb(torch.cat(output_diffs, 1)[0])
                output_masks = torch.nn.functional.interpolate(torch.cat(output_masks, 1), size=(h, w), mode='bilinear', align_corners=True)
                mask_out_left, mask_out_right = utils.network_utils.graybi2rgb(output_masks[0])
                disp_out_left, disp_out_right = utils.network_utils.graybi2rgb(output_disps[0])
                img_out_left = imgs_prd[0][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                img_out_right = imgs_prd[1][0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                test_writer.add_image('StereoDeblurNet/IMG_OUT' + str(batch_idx + 1), torch.cat([img_out_left, img_out_right], 2), epoch_idx + 1)
                test_writer.add_image('StereoDeblurNet/DISP_OUT'+str(batch_idx+1), torch.cat([disp_out_left, disp_out_right], 2), epoch_idx+1)
                test_writer.add_image('StereoDeblurNet/DIFF_OUT'+str(batch_idx+1), torch.cat([diff_out_left, diff_out_right], 2), epoch_idx+1)
                test_writer.add_image('StereoDeblurNet/MAST_OUT'+str(batch_idx+1), torch.cat([mask_out_left, mask_out_right], 2), epoch_idx+1)
            if cfg.NETWORK.PHASE == 'test':
                img_left_dir = os.path.join(cfg.DIR.OUT_PATH,'stereo',names[0],'left')
                img_right_dir = os.path.join(cfg.DIR.OUT_PATH,'stereo',names[0],'right')

                if not os.path.isdir(img_left_dir):
                    mkdir(img_left_dir)
                if not os.path.isdir(img_right_dir):
                    mkdir(img_right_dir)

                print(img_left_dir + '/' + str(save_num).zfill(4) + '.png')
                cv2.imwrite(img_left_dir + '/' + str(save_num).zfill(4) + '.png',
                            (imgs_prd[0].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                np.uint8),
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 5])

                print(img_right_dir + '/' + str(save_num).zfill(4) + '.png')
                cv2.imwrite(img_right_dir + '/' + str(save_num).zfill(4) + '.png',
                            (imgs_prd[1].clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 5])

    # Output testing results

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
        print('[TEST] [Epoch{0}]\t BatchTime_avg {1}\t DataTime_avg {2}\t  DispEPE_avg {3}\t ImgPSNR_avg {4}\n'
              .format(cfg.TRAIN.NUM_EPOCHES, batch_time.avg, data_time.avg, disp_EPEs.avg, img_PSNRs.avg))

        # Add testing results to TensorBoard
        test_writer.add_scalar('StereoDeblurNet/EpochEPE_1_TEST', disp_EPEs.avg, epoch_idx + 1)
        test_writer.add_scalar('StereoDeblurNet/EpochPSNR_1_TEST', img_PSNRs.avg, epoch_idx + 1)

        return disp_EPEs.avg, img_PSNRs.avg