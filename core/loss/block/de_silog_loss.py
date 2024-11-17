
import torch.nn as nn
import torch
from ..loss_builder import LOSS_BLOCK



@LOSS_BLOCK.register_module()
class Silog_loss(nn.Module):
    def __init__(self, silog_loss_weight, alpha_image_loss, depth_min_eval):
        self.alpha_image_loss = alpha_image_loss
        self.depth_min_eval = depth_min_eval
        self.silog_loss_weight = silog_loss_weight
        
    def forward(self, x):
        depth_est = x[0]
        depth_gt = x[1]
        mask = x[2]
        
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        self.depth_value = (d ** 2).mean()
        self.depth_value_2 = d.mean() ** 2
        
        depth_loss = torch.sqrt((d ** 2).mean() - self.alpha_image_loss * (d.mean() ** 2)) * 10.0
        return self.silog_loss_weight * depth_loss
    