
from local_configs.run_parser import MainParser
import yaml 
# import torch._dynamo

import argparse

import torch.multiprocessing as mp

from mmcv.utils import Config
# rebuttal before
from model_train import Model_Train
from model_test import Model_Test
from model_samples_test import Model_Samples_Test
from video_test import Model_Video_Test


        
def run_hyperparm_setting(opt, config):
    space1 = "".rjust(5)
    space2 = "".rjust(10)

    print("🚀 HyperParameters")
    for k, v in config.items():
        if isinstance(v, dict):
            print(space1 + f"{k}:")
            for k2, v2 in v.items():
                print(space2 + f"{k2}".ljust(20) + f"{v2}")
                opt.__dict__[k2] = v2
        else:
            opt.__dict__[k] = v
            print(space1 + f"{k}:".ljust(25) + f"{v}")

    return opt


def main(opt: argparse.Namespace):
    
    print("\n🚀🚀🚀 About the Parameters on this Project! 🚀🚀🚀")
    cfg = Config.fromfile(opt.config)

    opt = run_hyperparm_setting(opt, cfg)
    opt.log_comment = opt.config[:-3].split('/')[-1]
    
    if opt.mode == 'train':
        Model_Train(opt).train()
        
    elif opt.mode == 'test':
        Model_Test(opt).test()

    elif opt.mode == 'samples_test':
        Model_Samples_Test(opt).samples_test()

    elif opt.mode == 'video_test':
        Model_Video_Test(opt).video_test()
        
    
if __name__ == '__main__':    
    parser = MainParser()
    
    opt = parser.parse()
    main(opt)
