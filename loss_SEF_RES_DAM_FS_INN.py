#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

# 梯度损失：sobel算子
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

# 多尺度结构性损失
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()  

    def forward(self,image_vis,image_ir,generate_img):
  
        # l1loss计算两个张量之间每个元素的绝对误差的平均值 即对应位置元素差的绝对值的平均值
        # 强度损失或者像素损失
        x_in_max=torch.max(image_vis,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        
        #通过sobel算子计算梯度损失 梯度损失=loss_texture
        vis_grad=self.sobelconv(image_vis)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(vis_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        
        #多尺度结构性损失
        loss_structure_vis = 1 - msssim(generate_img, image_vis, normalize=True)
        loss_structure_ir = 1 - msssim(generate_img, image_ir, normalize=True)
        loss_msssim = loss_structure_ir + loss_structure_vis
        
        # 总损失
        loss_total = loss_in + 10*loss_grad + loss_msssim 
        
        return loss_total, loss_in, loss_grad, loss_msssim

    


