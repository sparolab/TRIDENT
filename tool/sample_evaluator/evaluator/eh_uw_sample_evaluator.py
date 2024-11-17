
from genericpath import isdir
import torch
import tqdm
import numpy as np
import torch.distributed as dist
import os
from PIL import Image
import math

from ..evaluate_utils.underwater.uiqm_utils import uiqm_compute
from ..evaluate_utils.underwater.uciqe_utils import uciqe_compute
import glob
from ..sample_evaluation_builder import SAMPLE_EVALUATOR_BUILDER, SAMPLE_EVALUATOR

@SAMPLE_EVALUATOR.register_module()
class Sample_UW_Enhancement_Evaluator(object):
    def __init__(self,
                 auto_crop,
                 img_size,
                 is_do_crop,
                 eval_dir,
                 gt_dir,
                 sample_eval_log_comment,
                 save_dir=None,
                 is_txt_save:bool=True
                 ):

        if is_txt_save is True:
            if save_dir is None:
                raise ValueError("If 'is_txt_save' is True, then 'save_dir' is must be not 'None'. but, Got {}".format(save_dir))
        
            if os.path.isdir(save_dir) is False:
                raise ValueError("'save dir' is not exist. but, Got {}".format(save_dir))
        
        
        self.eval_metrics = ['uw_abs_rel', 'uw_log10', 'uw_rms', 'uw_sq_rel', 'uw_log_rms', 'uw_uiqm', 'uw_uciqe' , 'uw_psnr']
        self.metrics_len = len(self.eval_metrics)
        
        self.save_dir = save_dir
        self.sample_eval_log_comment = 'uw_eh.' + sample_eval_log_comment
        self.is_txt_save = is_txt_save

        self.eval_dir = eval_dir
        self.gt_dir = gt_dir
        
        self.auto_crop = auto_crop
        self.img_size = img_size
        self.is_do_crop = is_do_crop
        
        
    def enhanced_compute_errors(self, gt, pred):
        rms = (gt - pred) ** 2
        rms = np.sqrt(rms.mean())

        log_rms = (np.log(gt) - np.log(pred)) ** 2
        log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        err = np.log(pred) - np.log(gt)
        log10 = np.mean(err)

        pred = pred * 255.0
        gt = gt * 255.0
        
        mse = np.mean( (gt - pred) ** 2 )
        psnr = 10 * math.log10(255.0**2/mse)
        
        uiqm = uiqm_compute(pred)
        uciqe = uciqe_compute(pred)

        return [abs_rel, log10, rms, sq_rel, log_rms, uiqm, uciqe, psnr]


    def enhanced_eval(self):
        space1 = " "*5
        
        num_metrics = self.metrics_len
        
        eval_measures = torch.zeros(num_metrics + 1)
        
        eval_list = glob.glob(self.eval_dir + '/*')
        eval_list.sort()

        gt_list = glob.glob(self.gt_dir + '/*')
        gt_list.sort()
        
        assert len(eval_list) == len(gt_list)
        
        print('len(eval_list): ', len(eval_list))
        
        for index in tqdm.tqdm(range(len(eval_list)), total=len(eval_list)):
            eval_data = eval_list[index]
            gt_data = gt_list[index]
            
            image = Image.open(eval_data)
            gt_enhanced = Image.open(gt_data)
            
            if self.auto_crop is True:
                auto_height = 32 * (image.height // 32)
                auto_width = 32 * (image.width // 32)   
                top_margin = int((image.height - auto_height) / 2) 
                left_margin = int((image.width - auto_width) / 2)             

                image = image.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height)) 
                gt_enhanced = gt_enhanced.crop((left_margin, top_margin, left_margin + auto_width, top_margin + auto_height))  
         
            else:
                if self.is_do_crop:
                    height = image.height
                    width = image.width
                    top_margin = int((height - self.img_size[0]) / 2) 
                    left_margin = int((width - self.img_size[1]) / 2)
                    
                    image = image.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))                  
                    gt_enhanced = gt_enhanced.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))
                
            image = np.asarray(image) / 255.0
            gt_enhanced = np.asarray(gt_enhanced) / 255.0
            
            gt_enhanced[gt_enhanced < 0.00001] = 0.00001
            
            measures = self.enhanced_compute_errors(gt_enhanced, image)

            eval_measures[:num_metrics] += torch.tensor(measures)
            eval_measures[num_metrics] += 1

        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[num_metrics].item()
        eval_measures_cpu /= cnt
        print(space1+'🚀 E.H: Computing errors for {} eval samples'.format(int(cnt)))

        error_list = []
        for i in range(num_metrics):
            error_string = '{} : {:.4f} '.format(self.eval_metrics[i], eval_measures_cpu[i])
            error_list.append(error_string)
            print(error_string)
        
        result = {'eval_measures': eval_measures_cpu, 'error_list': error_list}
        return result


    def evalutate_worker(self):
        space1 = " "*5
        
        result_commpute = self.enhanced_eval()
        
        eval_measures = result_commpute['eval_measures']
        error_list = result_commpute['error_list']

        filepath = os.path.join(self.save_dir, self.sample_eval_log_comment)
        
        if self.is_txt_save:
            if os.path.isfile(filepath):     
                os.remove(filepath)
            
            with open(filepath + '.txt', 'w') as f:
                for error_string in error_list:
                    f.write(error_string)
                    f.write('\n')
                    
            print(space1+'🚀 Successful saving')
        else:
            for error_string in error_list:
                print(error_string)
            

