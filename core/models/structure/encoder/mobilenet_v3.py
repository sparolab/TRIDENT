
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

from mmcv.utils import Registry
from ...network_builder import ENCODER

import numpy as np

import timm


# ENCODER  = Registry('encoder')

class MobileNetV3(nn.Module):
    def __init__(self, 
                 model_name='mobilenetv3_large', 
                 take_layer_name=['features'], 
                 skip_layer_name=['features1','features3','features6','features12', 'features15'],
                 skip_layer_output_channel=[16, 24, 40, 112, 160],
                 use_pretrained=True, 
                 use_hsv=False,
                 **kwargs):
        super().__init__()
        
        self.model_name = model_name
        self.take_layer_name = take_layer_name
        self.skip_layer_name = skip_layer_name
        self.skip_layer_output_channel = skip_layer_output_channel
        self.use_pretrained = use_pretrained
        self.weight = None
        self.use_hsv = use_hsv
        
        
        if self.model_name == 'mobilenetv3_small': 
            if self.use_pretrained is True:
                self.weight = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            
            model = models.mobilenet_v3_small(weights = self.weight)

        elif self.model_name == 'mobilenetv3_large':      
            if self.use_pretrained is True:
                self.weight = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            
            model = models.mobilenet_v3_large(weights = self.weight)
        else:
            raise ValueError("[User:INFO]: The model type is invalid!")

        
        if self.model_name == 'mobilenetv3_large':
            cutt_layer = 12

        elif self.model_name == 'mobilenetv3_small':
            cutt_layer = 8
        
        self.layer_name_list = []
        
        for layer_name, layer_modules in model._modules.items():
            if layer_name in self.take_layer_name:
                if layer_name == 'features':
                    for sub_layer_name, sub_layer_modules in layer_modules._modules.items():       
                        # print('------------------------------------------------------------------------')   
                        if int(sub_layer_name) <= cutt_layer: 
                            if self.use_hsv is True:
                                sub_layer_modules = nn.Sequential(
                                    nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                    nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                    nn.Hardswish()
                                )
                                self.use_hsv = False
                            self.add_module(layer_name + sub_layer_name, sub_layer_modules)
                            self.layer_name_list.append(layer_name + sub_layer_name)
                else:
                    if self.use_hsv is True:
                        sub_layer_modules = nn.Sequential(
                            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                            nn.Hardswish()
                        )
                        self.use_hsv = False
                        
                    self.add_module(layer_name, layer_modules)
                    self.layer_name_list.append(layer_name)                  
            else:
                continue
        
        print(self.layer_name_list)


    def forward(self, x):
        skip_feature = []
        
        for layer_name in self.layer_name_list:
            x = self._modules[layer_name](x)
            
            if layer_name in self.skip_layer_name:                
                skip_feature.append(x)     
        
        return skip_feature


@ENCODER.register_module()
class MobileNetV3_Small_NON32(MobileNetV3):
    def __init__(self, use_pretrained, **kwargs):
        super(MobileNetV3_Small_NON32, self).__init__(**kwargs, 
                                                model_name='mobilenetv3_small',
                                                take_layer_name=['features'], 
                                                skip_layer_name=['features0',   # (H/2, W/2)
                                                                'features1',    # (H/4, W/4)
                                                                'features3',    # (H/8, W/8)
                                                                'features8',    # (H/16, W/16)
                                                                ],
                                                skip_layer_output_channel=[16, 16, 24, 48],
                                                use_pretrained=use_pretrained
                                                )


@ENCODER.register_module()
class MobileNetV3_Small(MobileNetV3):
    def __init__(self, use_pretrained, **kwargs):
        super(MobileNetV3_Small, self).__init__(**kwargs, 
                                                model_name='mobilenetv3_small',
                                                take_layer_name=['features'], 
                                                skip_layer_name=['features0',   # (H/2, W/2)
                                                                'features1',    # (H/4, W/4)
                                                                'features3',    # (H/8, W/8)
                                                                'features8',    # (H/16, W/16)
                                                                'features11'    # (H/32, W/32)
                                                                ],
                                                skip_layer_output_channel=[16, 16, 24, 48, 96],
                                                use_pretrained=use_pretrained
                                                )

@ENCODER.register_module()
class MobileNetV3_Large_NON32(MobileNetV3):
    def __init__(self, use_pretrained, **kwargs):
        super(MobileNetV3_Large_NON32, self).__init__(**kwargs, 
                                                model_name='mobilenetv3_large',
                                                take_layer_name=['features'], 
                                                skip_layer_name=['features1',   # (H/2, W/2)
                                                                'features3',    # (H/4, W/4)
                                                                'features6',    # (H/8, W/8)
                                                                'features12',   # (H/16, W/16)
                                                                ],
                                                skip_layer_output_channel=[16, 24, 40, 112],
                                                use_pretrained=use_pretrained
                                                )
        
        
@ENCODER.register_module()
class MobileNetV3_Large_NON16(MobileNetV3):
    def __init__(self, use_pretrained, **kwargs):
        super(MobileNetV3_Large_NON16, self).__init__(**kwargs, 
                                                model_name='mobilenetv3_large',
                                                take_layer_name=['features'], 
                                                skip_layer_name=['features1',   # (H/2, W/2)
                                                                'features3',    # (H/4, W/4)
                                                                'features6',    # (H/8, W/8)
                                                                ],
                                                skip_layer_output_channel=[16, 24, 40],
                                                use_pretrained=use_pretrained
                                                )



@ENCODER.register_module()
class MobileNetV3_Large_NON16_HSV(MobileNetV3):
    def __init__(self, use_pretrained, **kwargs):
        super(MobileNetV3_Large_NON16_HSV, self).__init__(**kwargs, 
                                                model_name='mobilenetv3_large',
                                                take_layer_name=['features'], 
                                                skip_layer_name=['features1',   # (H/2, W/2)
                                                                'features3',    # (H/4, W/4)
                                                                'features6',    # (H/8, W/8)
                                                                ],
                                                skip_layer_output_channel=[16, 24, 40],
                                                use_pretrained=use_pretrained,
                                                use_hsv = True
                                                )



@ENCODER.register_module()
class MobileNetV3_Large(MobileNetV3):
    def __init__(self, use_pretrained, **kwargs):
        super(MobileNetV3_Large, self).__init__(**kwargs, 
                                                model_name='mobilenetv3_large',
                                                take_layer_name=['features'], 
                                                skip_layer_name=['features1',   # (H/2, W/2)
                                                                'features3',    # (H/4, W/4)
                                                                'features6',    # (H/8, W/8)
                                                                'features12',   # (H/16, W/16)
                                                                'features15'    # (H/32, W/32)
                                                                ],
                                                skip_layer_output_channel=[16, 24, 40, 112, 160],
                                                use_pretrained=use_pretrained
                                                )


                 
if __name__ == '__main__':
    tmp_cfg = dict(type='MobileNetV3_Large',
                use_pretrained=False
                )
       
    model = ENCODER.build(tmp_cfg)
    # print(model)
    model = model.to('cuda:0')

    result = model(torch.rand(1, 3, 224, 224).to('cuda:0'))
    
    iter_num = 100
    
    start = time.time()
    with torch.no_grad():  
        for a in tqdm.tqdm(range(iter_num)):
            result = model(torch.rand(1, 3, 224, 224).to('cuda:0'))
        elapsed_time = time.time()-start
    print("Elapesed time: '{} sec' for '{} files' -> '{} Hz'".format(str(elapsed_time), iter_num, iter_num/elapsed_time))
    
    num_params_update = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {} M".format(num_params_update / 1000000.0))
    num_params_update = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {} M".format(num_params_update / 1000000.0))