

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from ..loss_builder import LOSS_BLOCK


@LOSS_BLOCK.register_module()
class SSIMLoss(torch.nn.Module):
    def __init__(self, lambda_ssim, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
        self.lambda_ssim = lambda_ssim

    def forward(self, x):
        img1 = x[0]
        img2 = x[1]
        mask = x[2]
        
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
            
        loss = self.lambda_ssim * (1 - self._ssim(img1,
                                                  img2,
                                                  window,
                                                  self.window_size,
                                                  channel,
                                                  mask,
                                                  self.size_average))
        return loss

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, mask, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        ssim_map_red = ssim_map[:,0,:,:].unsqueeze(1)
        ssim_map_green = ssim_map[:,1,:,:].unsqueeze(1)
        ssim_map_blue = ssim_map[:,2,:,:].unsqueeze(1)
        
        ssim_map_loss = (ssim_map_red[mask] + ssim_map_green[mask] + ssim_map_blue[mask]).mean()/3.0
        return ssim_map_loss
        