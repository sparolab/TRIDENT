
import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tr

from core.models.network_builder import MODEL_BUILDER
from dataset.dataload_builder import DATALOAD_BUILDER
import tqdm

from matplotlib import pyplot as plt
from PIL import Image

from utils.image_processing import normalize_result, uw_inv_normalize
from utils.research_visualizing import AttentionMapVisualizing

class Model_Samples_Test(object):
    def __init__(self, opt):
        self.opt = opt
    
    def device_initialize(self, 
                          device='', 
                          batch_size=1):
        
        device = str(device).strip().lower().replace('cuda:', '').strip()  # to string, 'cuda:0' to '0'
        cpu = device == 'cpu'
        
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            torch.cuda.empty_cache()        # 언제나 GPU를 한번 씩 비워주자.
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
            assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
                f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

        cuda_flag = not cpu and torch.cuda.is_available()

        if cuda_flag:
            devices = device.split(',') if device else '0'
            space1 = ' ' * 5
            print(space1+ f"devices: {devices}")

            n = len(devices)
            if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
                assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
            space2 = ' ' * 10
            for i, d in enumerate(devices):
                p = torch.cuda.get_device_properties(i)
                print(f"{space2}🚀 CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)")  # bytes to MB
            
            return devices
        else:
            print('🚀 CPU is used!')
            return device        
        
    def get_num_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines) 


    def samples_test(self):
        """Test function."""
        space1 = " "*5 
        space2 = " "*10
        
        print("\n🚀🚀🚀 Setting Gpu before Test! 🚀🚀🚀")
        device =  self.device_initialize(device=self.opt.device, batch_size=self.opt.batch_size)

        print("\n🚀🚀🚀 Setting Model for Test!! 🚀🚀🚀")
        model = MODEL_BUILDER.build(self.opt.model_cfg)
                 
        if device != 'cpu':
            device = int(device[0])
            loc = 'cuda:{}'.format(device)
            checkpoint = torch.load(self.opt.test_checkpoint_path, map_location = loc)
            model.load_state_dict(checkpoint['model'])
            model.to('cuda:{}'.format(device))
        else:
            checkpoint = torch.load(self.opt.test_checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])      
            
        model.eval()    

        normalizer = tr.Normalize(mean=[0.13553666, 0.41034216, 0.34636855], std=[0.04927989, 0.10722694, 0.10722694])
        
        all_files_list = os.listdir(self.opt.sample_test_cfg['sample_data_path'])
        all_files_list.sort()
    
        print(space1+"🚀 now testing file name is '{}'".format(len(all_files_list)))
        print(space1+'🚀 Try to make directories')
        save_name = 'sample_eval_result_' + self.opt.log_comment
        save_enhanced_name = 'sample_eval_result_' + self.opt.log_comment + '/enhanced'
        save_depth_raw_name = 'sample_eval_result_' + self.opt.log_comment + '/depth_raw'
        save_depth_normalised_name = 'sample_eval_result_' + self.opt.log_comment + '/depth_color'
        save_uncert_raw_name = 'sample_eval_result_' + self.opt.log_comment + '/uncert_raw'
        save_uncert_normalised_name = 'sample_eval_result_' + self.opt.log_comment + '/uncert_color'
        
        os.mkdir(save_name)
        os.mkdir(save_enhanced_name)
        os.mkdir(save_depth_raw_name)
        os.mkdir(save_depth_normalised_name)
        
        if self.opt.is_triple_train is True:
            os.mkdir(save_uncert_raw_name)
            os.mkdir(save_uncert_normalised_name)
        
        prev_time = time.time() 
        total_frames = 0
        fps_list = []
        print("\n🚀🚀🚀 Start Test!! 🚀🚀🚀")
        print(space1+"🚀 Precdicting Inputs...")
        with torch.no_grad():
            for frame_name in tqdm.tqdm(all_files_list):
                
                data_path = os.path.join(self.opt.sample_test_cfg['sample_data_path'], frame_name)
                data = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) / 255.0
                
                height, width = data.shape[:2]
                
                if self.opt.sample_test_cfg['auto_crop'] is True:
                    revised_width = 32 * (width // 32)
                    revised_height = 32 * (height // 32)
                    
                    top_margin = int((height - revised_height) / 2) 
                    left_margin = int((width - revised_width) / 2)     

                    bottom_margin = int(top_margin + revised_height)
                    right_margin = int(left_margin + revised_width)
                    
                    do_resize_crop = False
                                    
                else:
                    revised_width = self.opt.sample_test_cfg['img_size'][1]
                    revised_height = self.opt.sample_test_cfg['img_size'][0]
                    revised_width = 32 * (revised_width // 32)
                    revised_height = 32 * (revised_height // 32)
                    
                    if self.opt.sample_test_cfg['do_resize_crop'] is True:
                        do_resize_crop = True
                    
                    else:
                        if self.opt.sample_test_cfg['do_center_crop'] is True:
                            top_margin = int((height - revised_height) / 2) 
                            left_margin = int((width - revised_width) / 2) 
                            
                            bottom_margin = int(top_margin + revised_height)
                            right_margin = int(left_margin + revised_width)
                            
                            do_resize_crop = False
    
                if do_resize_crop:
                    data = cv2.resize(data, (revised_width, revised_height), interpolation=cv2.INTER_CUBIC)
                else:
                    data = data[top_margin:bottom_margin, left_margin:right_margin, :]
                    
                img = torch.from_numpy(data.transpose((2, 0 ,1))).float()
                img = normalizer(img).unsqueeze(0)
                
                if self.opt.is_triple_train is True:
                    depth_est, _, uncertainty, enhanced_est, _, _ = model(img.cuda()) 
                else:
                    depth_est, _, enhanced_est, _, _ = model(img.cuda()) 
                    
                pred_enhanced = enhanced_est[0].cpu().numpy().transpose(1,2,0)
                pred_enhanced_scaled = pred_enhanced * 255.0
                pred_enhanced_scaled = pred_enhanced_scaled.astype(np.uint8) 
                Image.fromarray(pred_enhanced_scaled).save(os.path.join(save_enhanced_name, frame_name))

                attn_visualizer = AttentionMapVisualizing(head_fusion='mean', discard_ratio=0.8)

                pred_depth = depth_est.cpu().numpy().squeeze()
                pred_depth_scaled = pred_depth * float(1000)
                pred_depth_scaled = pred_depth_scaled.astype(np.uint16)         
                Image.fromarray(pred_depth_scaled).save(os.path.join(save_depth_raw_name, frame_name.replace('.jpg', '.png')))

                atten_mask_list = attn_visualizer([depth_est])
                
                image = attn_visualizer.show_mask(mask=1/np.array(atten_mask_list[0].cpu()), color=cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(save_depth_normalised_name, frame_name), image) 

                if self.opt.is_triple_train is True:
                    
                    uncertainty_visual = torch.mean(uncertainty, dim=1, keepdim=True).cpu().data
                    uncertainty_save = uncertainty_visual.squeeze().numpy()

                    uncertainty_scaled = uncertainty_save * 15.0
                    uncertainty_scaled = uncertainty_scaled.astype(np.uint16)
                    Image.fromarray(uncertainty_scaled).save(os.path.join(save_uncert_raw_name, frame_name.replace('.jpg', '.png')))

                    atten_mask_list = attn_visualizer([uncertainty_visual])
                    image = attn_visualizer.show_mask(mask=atten_mask_list[0], color=cv2.COLORMAP_MAGMA, uncert=0.6)
                    cv2.imwrite(os.path.join(save_uncert_normalised_name, frame_name), image) 
                

    