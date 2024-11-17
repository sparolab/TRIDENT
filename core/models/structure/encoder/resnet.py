
import torch
import torchvision.transforms as tr
from torchvision import models
import torch.functional as F
import torchsummaryX
import torch.nn as nn
from functools import partial
import math

import os

from mmcv.utils import Registry
from ...network_builder import ENCODER



class ResNet(nn.Module):
    def __init__(self, 
                 model_name='resnet101', 
                 skip_layer_name=['layer1', 'layer2', 'layer3', 'layer4'], 
                 use_pretrained=True, 
                 **kwargs):
        super().__init__()
        
        self.model_name = model_name
        self.skip_layer_name = skip_layer_name
        self.use_pretrained = use_pretrained
        self.weight = None
        
        
        if self.model_name == 'resnet18':
            if self.use_pretrained is True:
                self.weight = models.ResNet18_Weights.IMAGENET1K_V1
            
            model = models.resnet18(weights=self.weight)


        elif self.model_name == 'resnet34':
            if self.use_pretrained is True:
                self.weight = models.ResNet34_Weights.IMAGENET1K_V1
            
            model = models.resnet34(weights=self.weight)
   
   
        elif self.model_name == 'resnet50':
            if self.use_pretrained is True:
                self.weight = models.ResNet50_Weights.IMAGENET1K_V2
            
            model = models.resnet50(weights=self.weight)


        elif self.model_name == 'resnet101':
            if self.use_pretrained is True:
                self.weight = models.ResNet101_Weights.IMAGENET1K_V2
            
            model = models.resnet101(weights=self.weight)
            self.output_channel = [256, 512, 1024, 2048]


        elif self.model_name == 'resnet152':
            if self.use_pretrained is True:
                self.weight = models.ResNet152_Weights.IMAGENET1K_V2
            
            model = models.resnet152(weights=self.weight)
            self.output_channel = [256, 512, 1024, 2048]

        else:
            raise ValueError("[User:INFO]: The model type is invalid!")
   
        
        self.layer_name_list = []
        for layer_name, layer_modules in model._modules.items():
            if 'fc' in layer_name or 'avgpool' in layer_name:
                continue
            
            self.add_module(layer_name, layer_modules)
            self.layer_name_list.append(layer_name)
        
        # print(self.layer_name_list)


    def forward(self, x):
        
        index = 0
        skip_feature = []
        
        for layer_name in self.layer_name_list:
            x = self._modules[layer_name](x)
            
            if layer_name == self.skip_layer_name[index]:
                
                skip_feature.append(x)
                index = index +1
        
        print(skip_feature[0].shape)
        print(skip_feature[1].shape)
        print(skip_feature[2].shape)
        print(skip_feature[3].shape)
        
        return skip_feature



@ENCODER.register_module()
class ResNet18(ResNet):
    def __init__(self, use_pretrained, **kwargs):
        super(ResNet18, self).__init__(**kwargs, 
                                       model_name='resnet18',
                                       skip_layer_name=['layer1', 'layer2', 'layer3', 'layer4'],
                                       use_pretrained=use_pretrained
                                       )

@ENCODER.register_module()
class ResNet34(ResNet):
    def __init__(self, use_pretrained, **kwargs):
        super(ResNet34, self).__init__(**kwargs, 
                                       model_name='resnet34',
                                       skip_layer_name=['layer1', 'layer2', 'layer3', 'layer4'],
                                       use_pretrained=use_pretrained
                                       )


@ENCODER.register_module()
class ResNet50(ResNet):
    def __init__(self, use_pretrained, **kwargs):
        super(ResNet50, self).__init__(**kwargs, 
                                       model_name='resnet50',
                                       skip_layer_name=['layer1', 'layer2', 'layer3', 'layer4'],
                                       use_pretrained=use_pretrained
                                       )

@ENCODER.register_module()
class ResNet101(ResNet):
    def __init__(self, use_pretrained, **kwargs):
        super(ResNet101, self).__init__(**kwargs, 
                                        model_name='resnet101',
                                        skip_layer_name=['layer1', 'layer2', 'layer3', 'layer4'],
                                        use_pretrained=use_pretrained
                                        )

@ENCODER.register_module()
class ResNet152(ResNet):
    def __init__(self, use_pretrained, **kwargs):
        super(ResNet152, self).__init__(**kwargs, 
                                        model_name='resnet152',
                                        skip_layer_name=['layer1', 'layer2', 'layer3', 'layer4'],
                                        use_pretrained=use_pretrained
                                        )


tmp_cfg = dict(type='ResNet152',
               use_pretrained=True
               )
                 


if __name__ == '__main__':
    model = ENCODER.build(tmp_cfg)
    
    model = model.to('cuda:0')
    
    torchsummaryX.summary(model, torch.rand(2, 3, 480, 640).to('cuda:0'))
    
    
    
