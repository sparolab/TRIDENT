

import torch
import torchvision.transforms as tr
import torch.functional as F
import torchsummaryX
import torch.nn as nn
import warnings

from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import math

from mmcv.utils import Registry
import os

SAMPLE_EVALUATOR = Registry('sample_evaluator')
SAMPLE_EVALUATOR_BUILDER = Registry('sample_evaluator_builder')


@SAMPLE_EVALUATOR_BUILDER.register_module()
class Build_Sample_Evaluator(object):
    def __init__(self, evaluator_cfg_list:list, save_dir, sample_eval_log_comment):
        super().__init__()
                 
        self.evaluator_list = []
        for evaluator_cfg in evaluator_cfg_list:
            evaluator_cfg['save_dir'] = save_dir
            evaluator_cfg['sample_eval_log_comment'] = sample_eval_log_comment
            
            self.evaluator_list.append(SAMPLE_EVALUATOR.build(evaluator_cfg))    
    
    def result_evaluation(self):
        final_commpute = []
        
        for evaluator in self.evaluator_list:
            final_commpute.append(evaluator.evalutate_worker())
        
        return final_commpute
        
    