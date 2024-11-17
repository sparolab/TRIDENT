
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

class Model_Video_Test(object):
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


    def video_test(self):
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
        
        cap = cv2.VideoCapture(self.opt.video_test_cfg['video_txt_file'])
        origin_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        origin_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        
        if self.opt.video_test_cfg['auto_crop'] is True:
            revised_width = 32 * (origin_width // 32)
            revised_height = 32 * (origin_height // 32)
            
            top_margin = int((origin_height - revised_height) / 2) 
            left_margin = int((origin_width - revised_width) / 2)     

            bottom_margin = int(top_margin + revised_height)
            right_margin = int(left_margin + revised_width)
            
            do_resize_crop = False
        else:
            revised_width = self.opt.video_test_cfg['img_size'][1]
            revised_height = self.opt.video_test_cfg['img_size'][0]
            revised_width = 32 * (revised_width // 32)
            revised_height = 32 * (revised_height // 32)
            
            if self.opt.video_test_cfg['do_resize_crop'] is True:
                do_resize_crop = True
            
            else:
                if self.opt.video_test_cfg['do_center_crop'] is True:
                    top_margin = int((origin_height - revised_height) / 2) 
                    left_margin = int((origin_width - revised_width) / 2) 
                    
                    bottom_margin = int(top_margin + revised_height)
                    right_margin = int(left_margin + revised_width)
                    
                    do_resize_crop = False
                      

        print(space1+"🚀 now testing file name is '{}'".format(self.opt.video_test_cfg['video_txt_file']))
        print(space1+'🚀 Try to make directories')
        save_name = 'video_result_' + self.opt.log_comment

        prev_time = time.time() 
        total_frames = 0
        pred_enhanced = []
        pred_depth = []
        fps_list = []
        print("\n🚀🚀🚀 Start Test!! 🚀🚀🚀")
        print(space1+"🚀 Precdicting Inputs...")
        with torch.no_grad():
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break            
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                                
                if do_resize_crop:
                    frame = cv2.resize(frame, (revised_width, revised_height), interpolation=cv2.INTER_CUBIC)
                else:
                    frame = frame[top_margin:bottom_margin, left_margin:right_margin, :]
                    
                img = torch.from_numpy(frame.transpose((2, 0 ,1))).float()
                img = normalizer(img).unsqueeze(0)
                
                if self.opt.is_triple_train is True:
                    depth_est, depth_scale_list, uncertainty, enhanced_est, forw_tmp_list, backw_tmp_list = model(img.cuda()) 

                else:
                    depth_est, depth_scale_list, enhanced_est, forw_tmp_list, backw_tmp_list = model(img.cuda()) 

                pred_enhanced = enhanced_est[0].cpu().numpy().transpose(1,2,0)

                current_time = time.time()       
                elapsed_time = current_time - prev_time
                prev_time = current_time
                fps = 1/(elapsed_time)
                str = "FPS: %0.2f" % fps
                pred_enhanced = cv2.cvtColor(pred_enhanced, cv2.COLOR_RGB2BGR)
                
                cv2.putText(pred_enhanced, str, (revised_width//20, revised_height//10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
                cv2.imshow("enhanced_frame", pred_enhanced)
                cv2.waitKey(1)
        
            cap.release()
            cv2.destroyAllWindows()