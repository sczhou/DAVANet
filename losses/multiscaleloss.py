import torch
import torch.nn as nn
from config import cfg
from utils.network_utils import *
#
# Disparity Loss
#
def EPE(output, target, occ_mask):
    N = torch.sum(occ_mask)
    d_diff = output - target
    EPE_map = torch.abs(d_diff)
    EPE_map = torch.mul(EPE_map, occ_mask)

    EPE_mean = torch.sum(EPE_map)/N
    return EPE_mean

def multiscaleLoss(outputs, target, img, occ_mask, weights):

    def one_scale(output, target, occ_mask):
        b, _, h, w = output.size()
        occ_mask = nn.functional.adaptive_max_pool2d(occ_mask, (h, w))
        if cfg.DATASET.SPARSE:
            target_scaled = nn.functional.adaptive_max_pool2d(target, (h, w))
        else:
            target_scaled = nn.functional.adaptive_avg_pool2d(target, (h, w))
        return EPE(output, target_scaled, occ_mask)

    if type(outputs) not in [tuple, list]:
        outputs = [outputs]

    assert(len(weights) == len(outputs))

    loss = 0
    for output, weight in zip(outputs, weights):
        loss += weight * one_scale(output, target, occ_mask)
    return loss

def realEPE(output, target, occ_mask):
    b, _, h, w = target.size()
    upsampled_output = nn.functional.interpolate(output, size=(h,w), mode = 'bilinear', align_corners=True)
    return EPE(upsampled_output, target, occ_mask)

#
# Deblurring Loss
#
def mseLoss(output, target):
    mse_loss = nn.MSELoss(reduction ='elementwise_mean')
    MSE = mse_loss(output, target)
    return MSE

def PSNR(output, target, max_val = 1.0):
    output = output.clamp(0.0,1.0)
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return torch.Tensor([100.0])
    return 10 * torch.log10(max_val**2 / mse)


def perceptualLoss(fakeIm, realIm, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''

    weights = [1, 0.2, 0.04]
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm)
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss
