
from genericpath import isdir
import torch
import tqdm
import numpy as np
import torch.distributed as dist
import os
from PIL import Image

import glob
from ..sample_evaluation_builder import SAMPLE_EVALUATOR_BUILDER, SAMPLE_EVALUATOR

@SAMPLE_EVALUATOR.register_module()
class Sample_DepthEstimation_Evaluator(object):
    def __init__(self,
                 min_depth_eval, 
                 max_depth_eval,
                 depth_scaling,
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
        
        
        self.eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
        self.metrics_len = len(self.eval_metrics)
        
        self.save_dir = save_dir
        self.sample_eval_log_comment = 'de.' + sample_eval_log_comment
        self.is_txt_save = is_txt_save

        self.min_depth_eval = min_depth_eval
        self.max_depth_eval = max_depth_eval
        self.depth_scaling = depth_scaling

        self.eval_dir = eval_dir
        self.gt_dir = gt_dir
        
        self.img_size = img_size
        self.is_do_crop = is_do_crop
        
        
    def depth_compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()

        rms = (gt - pred) ** 2
        rms = np.sqrt(rms.mean())

        log_rms = (np.log(gt) - np.log(pred)) ** 2
        log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(pred) - np.log10(gt))
        log10 = np.mean(err)

        return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


    def depth_eval(self):
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
            depth_scaling = self.depth_scaling
            
            depth_image = Image.open(eval_data)
            gt_depth = Image.open(gt_data)
            
            if self.is_do_crop:
                height = depth_image.height
                width = depth_image.width
                top_margin = int((height - self.img_size[0]) / 2) 
                left_margin = int((width - self.img_size[1]) / 2)
                
                depth_image = depth_image.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))                  
                gt_depth = gt_depth.crop((left_margin, top_margin, left_margin + self.img_size[1], top_margin + self.img_size[0]))
                
            depth_image = np.asarray(depth_image) / float(depth_scaling)
            depth_image = np.expand_dims(depth_image, axis= 0)
            
            gt_depth = np.asarray(gt_depth)
            gt_depth = np.expand_dims(gt_depth, axis= 0) / float(depth_scaling)
            
            depth_image[depth_image < self.min_depth_eval] = self.min_depth_eval
            depth_image[depth_image > self.max_depth_eval] = self.max_depth_eval
            
            depth_image[np.isnan(depth_image)] = self.min_depth_eval
            depth_image[np.isinf(depth_image)] = self.max_depth_eval
            
            valid_mask = np.logical_and(gt_depth > self.min_depth_eval, gt_depth < self.max_depth_eval)
            
            measures = self.depth_compute_errors(gt_depth[valid_mask], depth_image[valid_mask])

            eval_measures[:num_metrics] += torch.tensor(measures)
            eval_measures[num_metrics] += 1

        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[num_metrics].item()
        eval_measures_cpu /= cnt
        print(space1+'🚀 D.E: Computing errors for {} eval samples'.format(int(cnt)))

        error_list = []
        for i in range(num_metrics):
            error_string = '{}: {:.4f} '.format(self.eval_metrics[i], eval_measures_cpu[i])
            error_list.append(error_string)
            print(error_string)
        
        result = {'eval_measures': eval_measures_cpu, 'error_list': error_list}
        return result
    
    def evalutate_worker(self):
        space1 = " "*5
        
        result_commpute = self.depth_eval()
        
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
            

