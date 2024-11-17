
import torch
import torchvision.transforms as tr
from torchvision import models
import torch.functional as F
import torchsummaryX
import torch.nn as nn
from functools import partial
import math
import time
import tqdm
import os

import numpy as np

from mmcv.utils import Registry
# from ...network_builder import ENCODER

import timm.models

ENCODER  = Registry('encoder')

class EfficientFormer(nn.Module):
    def __init__(self, 
                 model_name='efficientformer_v2_s2', 
                 take_layer_name=['stem', 'stages'], 
                 skip_layer_name=['stages0','stages1','stages2','stages3'],
                 use_pretrained=True, 
                 **kwargs):
        super().__init__()
        
        self.model_name = model_name
        self.take_layer_name = take_layer_name
        self.skip_layer_name = skip_layer_name
        self.use_pretrained = use_pretrained
        self.weight = None
        
        
        if self.model_name == 'efficientformer_v2_s0': 
            model = timm.models.efficientformerv2_s0(pretrained = self.use_pretrained)

        elif self.model_name == 'efficientformer_v2_s1':
            model = timm.models.efficientformerv2_s1(pretrained = self.use_pretrained)
   
        elif self.model_name == 'efficientformer_v2_s2':
            model = timm.models.efficientformerv2_s2(pretrained = self.use_pretrained)

        elif self.model_name == 'efficientformer_v2_l':      
            model = timm.models.efficientformerv2_l(pretrained = self.use_pretrained)
        else:
            raise ValueError("[User:INFO]: The model type is invalid!")

        
        self.layer_name_list = []
        
        # print(model)
        
        for layer_name, layer_modules in model._modules.items():
            
            if layer_name in self.take_layer_name:
                if layer_name == 'stages':
                    for sub_layer_name, sub_layer_modules in layer_modules._modules.items():           
                        self.add_module(layer_name + sub_layer_name, sub_layer_modules)
                        self.layer_name_list.append(layer_name + sub_layer_name)
                else:
                    self.add_module(layer_name, layer_modules)
                    self.layer_name_list.append(layer_name)                  
            else:
                continue
        
        print(self.layer_name_list)


    def forward(self, x):
        
        index = 0
        skip_feature = []
        
        for layer_name in self.layer_name_list:
            x = self._modules[layer_name](x)
            
            if layer_name == self.skip_layer_name[index]:
                
                skip_feature.append(x)
                index = index +1
        
        return skip_feature



@ENCODER.register_module()
class EfficientFormerV2_S0(EfficientFormer):
    def __init__(self, use_pretrained, **kwargs):
        super(EfficientFormerV2_S0, self).__init__(**kwargs, 
                                                   model_name='efficientformer_v2_s0',
                                                   take_layer_name=['stem', 'stages'], 
                                                   skip_layer_name=['stages0','stages1','stages2','stages3'],
                                                   use_pretrained=use_pretrained
                                                   )


@ENCODER.register_module()
class EfficientFormerV2_S1(EfficientFormer):
    def __init__(self, use_pretrained, **kwargs):
        super(EfficientFormerV2_S1, self).__init__(**kwargs, 
                                                   model_name='efficientformer_v2_s1',
                                                   take_layer_name=['stem', 'stages'], 
                                                   skip_layer_name=['stages0','stages1','stages2','stages3'],
                                                   use_pretrained=use_pretrained
                                                   )

@ENCODER.register_module()
class EfficientFormerV2_S2(EfficientFormer):
    def __init__(self, use_pretrained, **kwargs):
        super(EfficientFormerV2_S2, self).__init__(**kwargs, 
                                                   model_name='efficientformer_v2_s2',
                                                   take_layer_name=['stem', 'stages'], 
                                                   skip_layer_name=['stages0','stages1','stages2','stages3'],
                                                   use_pretrained=use_pretrained
                                                   )

@ENCODER.register_module()
class EfficientFormerV2_L(EfficientFormer):
    def __init__(self, use_pretrained, **kwargs):
        super(EfficientFormerV2_L, self).__init__(**kwargs, 
                                                  model_name='efficientformer_v2_l',
                                                  take_layer_name=['stem', 'stages'], 
                                                  skip_layer_name=['stages0','stages1','stages2','stages3'],
                                                  use_pretrained=use_pretrained
                                                  )


tmp_cfg = dict(type='EfficientFormerV2_S0',
               use_pretrained=False
               )
                 

if __name__ == '__main__':
    
    model = ENCODER.build(tmp_cfg)
    model = model.to('cuda:0')

    iter_num = 1000
    result = model(torch.rand(1, 3, 224, 224).to('cuda:0'))
    
    start = time.time()
    with torch.no_grad():  
        for a in tqdm.tqdm(range(iter_num)):
            result = model(torch.rand(1, 3, 224, 224).to('cuda:0'))
        elapsed_time = time.time()-start
    print("Elapesed time: '{} sec' for '{} files' -> '{} Hz'".format(str(elapsed_time), iter_num, iter_num/elapsed_time))
    torchsummaryX.summary(model, torch.rand(2, 3, 224, 224).to('cuda:0'))
    
    
    num_params_update = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {} M".format(num_params_update / 1000000.0))
    num_params_update = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {} M".format(num_params_update / 1000000.0))
