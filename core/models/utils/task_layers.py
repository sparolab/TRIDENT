

import torchvision.transforms as tr
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..network_builder import TASK


@TASK.register_module()
class DepthEstimation_Task(nn.Module):
    def __init__(self, max_depth):
        super(DepthEstimation_Task, self).__init__()
        
        self.max_depth = max_depth
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)*self.max_depth
        return x


@TASK.register_module()
class Enhancement_Task(nn.Module):
    def __init__(self):
        super(Enhancement_Task, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)
        return x