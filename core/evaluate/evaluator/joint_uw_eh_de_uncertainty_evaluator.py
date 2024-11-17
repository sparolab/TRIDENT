
import torch
import tqdm
import numpy as np
import torch.distributed as dist
import os
import math

from ..evaluation_builder import EVALUATOR
import torchvision.transforms as tr


def inv_normalize(image):
    inv_normal = tr.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normal(image).data

def uw_inv_normalize(image):
    inv_normal = tr.Normalize(
        mean=[-0.13553666/0.04927989, -0.41034216/0.10722694, -0.34636855/0.10722694],
        std=[1/0.04927989, 1/0.10722694, 1/0.10722694]
    )
    return inv_normal(image).data


def normalize_result(value, vmin=None, vmax=None):
    
    try:
        value = value.cpu().numpy()[0, :, :]
    except:
        pass

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.
    return np.expand_dims(value, 0)



@EVALUATOR.register_module()
class Joint_UW_En_DE_Ucertain_Evaluator(object):
    def __init__(self,  
                 min_depth_eval, 
                 max_depth_eval,
                 device,
                 dataloader_eval,
                 ngpus:int=0,
                 save_dir=None,
                 is_checkpoint_save:bool=True,
                 is_triple_train:bool=False
                 ):

        if is_checkpoint_save is True:
            if save_dir is None:
                raise ValueError("If 'is_checkpoint_save' is True, then 'save_dir' is must be not 'False'. but, Got {}".format(save_dir))
        
            if os.path.isdir(save_dir) is False:
                raise ValueError("'save dir' is not exist. but, Got {}".format(save_dir))
        
        self.eval_metrics = ['uncertainty_loss', 'uw_abs_rel', 'uw_rms', 'uw_sq_rel', 'de_silog', 'de_abs_rel', 'de_log10', 'de_rms', 'de_sq_rel', 'de_log_rms', 
                             'uw_psnr', 'de_d1', 'de_d2', 'de_d3']
        
        num = len(self.eval_metrics)
        self.split_num = 10
        
        self.metrics_len = len(self.eval_metrics)
        
        self.is_checkpoint_save = is_checkpoint_save
        self.checkpoint_dir = save_dir
        
        self.is_triple_train = is_triple_train

        self.min_depth_eval = min_depth_eval
        self.max_depth_eval = max_depth_eval
        
        self.device = device
        self.ngpus = ngpus
        self.dataloader_eval = dataloader_eval

        self.best_eval_measures_lower_better = torch.zeros(self.split_num).cpu() + 1e4
        self.best_eval_measures_higher_better = torch.zeros(self.metrics_len - self.split_num).cpu()
        self.best_eval_steps = np.zeros(num, dtype= np.int32)
        
        self.peeking_num = 0


    def enhancement_and_depthestimation_compute_errors(self, enhanced_gt, enhanced_pred, depth_gt, depth_pred, uncertainty):

        ########### uncertainty estimation ###########
        if self.is_triple_train is True:
            uncertainty_gt_tmp = np.abs(depth_pred - depth_gt) / self.max_depth_eval
            uncertainty_loss = 512.0 * np.mean(np.abs(uncertainty-uncertainty_gt_tmp)**2)
        else:
            uncertainty_loss = 1.0

        ########### enhancement ###########
        uw_rms = (enhanced_gt - enhanced_pred) ** 2
        uw_rms = np.sqrt(uw_rms.mean())

        uw_abs_rel = np.mean(np.abs(enhanced_gt - enhanced_pred) / enhanced_gt)
        uw_sq_rel = np.mean(((enhanced_gt - enhanced_pred) ** 2) / enhanced_gt)

        enhanced_pred = enhanced_pred.transpose(1,2,0) * 255.0
        enhanced_gt = enhanced_gt.transpose(1,2,0) * 255.0

        uw_mse = np.mean((enhanced_gt - enhanced_pred) ** 2)
        uw_psnr = 10 * math.log10(255.0**2/uw_mse)
    

        ########### depth estimation ###########
        thresh = np.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()

        depth_rms = (depth_gt - depth_pred) ** 2
        depth_rms = np.sqrt(depth_rms.mean())

        depth_log_rms = (np.log(depth_gt) - np.log(depth_pred)) ** 2
        depth_log_rms = np.sqrt(depth_log_rms.mean())

        depth_abs_rel = np.mean(np.abs(depth_gt - depth_pred) / depth_gt)
        depth_sq_rel = np.mean(((depth_gt - depth_pred) ** 2) / depth_gt)

        err = np.log(depth_pred) - np.log(depth_gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(depth_pred) - np.log10(depth_gt))
        depth_log10 = np.mean(err)
        
    

        return [uncertainty_loss, uw_abs_rel, uw_rms, uw_sq_rel, silog, depth_abs_rel, depth_log10, depth_rms, 
                depth_sq_rel, depth_log_rms, uw_psnr, d1, d2, d3]


    def enhanced_eval(self, opt, model):
        space1 = " "*5
        
        num_metrics = self.metrics_len
        
        if self.device != None:
            eval_measures = torch.zeros(num_metrics + 1).cuda(device=self.device)
        else:
            eval_measures = torch.zeros(num_metrics + 1)
        
        self.val_image_sample = []
        self.val_scalar_sample = []
        for idx, eval_sample_batched in tqdm.tqdm(enumerate(self.dataloader_eval.data), 
                                                total=len(self.dataloader_eval.data)) if opt.rank == 0 or not opt.multiprocessing_distributed else enumerate(self.dataloader_eval.data):
            with torch.no_grad():
                image = torch.autograd.Variable(eval_sample_batched['image'].cuda(self.device, non_blocking=True))

                if self.is_triple_train is True:
                    depth_est, depth_scale_list, uncertainty, second_predicted, _, _ = model(image)
                else:
                    depth_est, depth_scale_list, second_predicted, _, _ = model(image)
                depth_gt = eval_sample_batched['depth']
                depth_8x8_gt = torch.nn.functional.interpolate(depth_gt, scale_factor=1/8, mode='nearest')
                depth_8x8_scaled_gt = torch.nn.functional.interpolate(depth_8x8_gt, scale_factor=8, mode='nearest') / opt.max_depth
                depth_8x8_scaled_gt = depth_8x8_scaled_gt.cpu().numpy().squeeze()
                
                depth_4x4_gt = torch.nn.functional.interpolate(depth_gt, scale_factor=1/4, mode='nearest')
                depth_4x4_scaled_gt = torch.nn.functional.interpolate(depth_4x4_gt, scale_factor=4, mode='nearest') / opt.max_depth
                depth_4x4_scaled_gt = depth_4x4_scaled_gt.cpu().numpy().squeeze()
                
                depth_2x2_gt = torch.nn.functional.interpolate(depth_gt, scale_factor=1/2, mode='nearest')       
                depth_2x2_scaled_gt = torch.nn.functional.interpolate(depth_2x2_gt, scale_factor=2, mode='nearest') / opt.max_depth
                depth_2x2_scaled_gt = depth_2x2_scaled_gt.cpu().numpy().squeeze()  

                depth_gt = depth_gt.cpu().numpy().squeeze()
                
                enhanced_gt = eval_sample_batched['enhanced'].cpu().numpy().squeeze()
                second_predicted = second_predicted.cpu().numpy().squeeze()
                depth_gt = np.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)

                
                depth_est = depth_est.cpu().numpy().squeeze()
                
                depth_8x8_scaled = depth_scale_list[0].cpu().numpy().squeeze()
                depth_4x4_scaled = depth_scale_list[1].cpu().numpy().squeeze()
                depth_2x2_scaled = depth_scale_list[2].cpu().numpy().squeeze()
                
                if self.is_triple_train is True:
                    uncertainty = uncertainty.cpu().numpy().squeeze()
         
          
            if idx == self.peeking_num:
                self.val_image_sample.append(uw_inv_normalize(image[0]))
                self.val_image_sample.append(second_predicted)
                self.val_image_sample.append(enhanced_gt)
                
                self.val_image_sample.append(normalize_result(1/depth_est))
                self.val_image_sample.append(normalize_result(1/depth_gt))

                self.val_image_sample.append(normalize_result(1/depth_8x8_scaled))
                self.val_image_sample.append(normalize_result(1/depth_4x4_scaled))
                self.val_image_sample.append(normalize_result(1/depth_2x2_scaled))

                if self.is_triple_train is True:
                    self.val_image_sample.append(normalize_result(uncertainty))

                self.val_image_sample.append(normalize_result(1/depth_8x8_scaled_gt))
                self.val_image_sample.append(normalize_result(1/depth_4x4_scaled_gt))
                self.val_image_sample.append(normalize_result(1/depth_2x2_scaled_gt))

            depth_est[depth_est < self.min_depth_eval] = self.min_depth_eval
            depth_est[depth_est > self.max_depth_eval] = self.max_depth_eval
            
            depth_est[np.isnan(depth_est)] = self.min_depth_eval
            depth_est[np.isinf(depth_est)] = self.max_depth_eval
            
            valid_mask = np.logical_and(depth_gt > self.min_depth_eval, depth_gt < self.max_depth_eval)
            
            if self.is_triple_train is False:
                uncertainty = depth_est
            measures = self.enhancement_and_depthestimation_compute_errors(enhanced_gt, second_predicted, depth_gt[valid_mask], depth_est[valid_mask], uncertainty[valid_mask])

            eval_measures[:num_metrics] += torch.tensor(measures).cuda(device=self.device)
            eval_measures[num_metrics] += 1

        if self.peeking_num == len(self.dataloader_eval.data):
            self.peeking_num = 0
        else:
            self.peeking_num = self.peeking_num + 1
        
        
        if opt.multiprocessing_distributed:
            group = dist.new_group([i for i in range(self.ngpus)])
            dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

        if not opt.multiprocessing_distributed or self.device == 0:
            eval_measures_cpu = eval_measures.cpu()
            cnt = eval_measures_cpu[num_metrics].item()
            eval_measures_cpu /= cnt
            print(space1+'🚀 E.H: Computing errors for {} eval samples'.format(int(cnt)))

            error_string = ''
            for i in range(num_metrics):
                error_string += '{}:{:.4f} '.format(self.eval_metrics[i], eval_measures_cpu[i])
            print(space1 + error_string)
        
        result = {'eval_measures': eval_measures_cpu, 'error_string': error_string}
        return result


    def check_best_eval_lower_better(self,
                                     metric, 
                                     eval_measures, 
                                     best_eval_measures_lower_better, 
                                     best_eval_steps, 
                                     global_step, 
                                     ):
        
        is_best = False
        if eval_measures < best_eval_measures_lower_better:
            old_best = best_eval_measures_lower_better.item()
            best_eval_measures_lower_better = eval_measures.item()
            is_best = True

        if is_best:
            old_best_step = best_eval_steps
            old_best_name = '/de_eh_model-{}-best_{}_{:.5f}.pth'.format(old_best_step, metric, old_best)
            model_path = self.checkpoint_dir + old_best_name
            if os.path.exists(model_path):
                command = 'rm {}'.format(model_path)
                os.system(command)
            best_eval_steps = global_step
            model_save_name = '/de_eh_model-{}-best_{}_{:.5f}.pth'.format(global_step, metric, eval_measures)
            print('E.H: New best for {}.'.format(model_save_name))
            
            result = {'best_eval_measures_lower_better':best_eval_measures_lower_better, 
                    'model_save_name': model_save_name, 
                    'best_eval_steps': best_eval_steps}
            return result
        else:
            result = None
            return result  


    def check_best_eval_higher_better(self,
                                      metric, 
                                      eval_measures, 
                                      best_eval_measures_higher_better, 
                                      best_eval_steps, 
                                      global_step
                                      ): 
    
        is_best = False
        if eval_measures > best_eval_measures_higher_better:
            old_best = best_eval_measures_higher_better.item()
            best_eval_measures_higher_better = eval_measures.item()
            is_best = True

        if is_best:
            old_best_step = best_eval_steps
            old_best_name = '/de_eh_model-{}-best_{}_{:.5f}.pth'.format(old_best_step, metric, old_best)
            model_path = self.checkpoint_dir + old_best_name
            if os.path.exists(model_path):
                command = 'rm {}'.format(model_path)
                os.system(command)
            
            best_eval_steps = global_step
            model_save_name = '/de_eh_model-{}-best_{}_{:.5f}.pth'.format(global_step, metric, eval_measures)
            print('E.H: New best for {}.'.format(model_save_name))
            
            result = {'best_eval_measures_higher_better':best_eval_measures_higher_better, 
                      'model_save_name':model_save_name, 
                      'best_eval_steps':best_eval_steps}
            return result
        else:
            result = None
            return result 
    
    
    def evalutate_worker(self, opt, model, global_step):
        
        result_commpute = self.enhanced_eval(opt, model)
        
        eval_measures = result_commpute['eval_measures']
        error_string = result_commpute['error_string']
        
        loss_list = []
        
        for idx in range(self.metrics_len):
            loss_list.append(eval_measures[idx])
            
            if idx < self.split_num:
                result = self.check_best_eval_lower_better(self.eval_metrics[idx],
                                                           eval_measures[idx], 
                                                           self.best_eval_measures_lower_better[idx], 
                                                           self.best_eval_steps[idx], 
                                                           global_step
                                                           )
                if result != None:
                    self.best_eval_measures_lower_better[idx] = result['best_eval_measures_lower_better']
                    model_save_name = result['model_save_name']
                    self.best_eval_steps[idx] = result['best_eval_steps']
            
            elif idx >= self.split_num:
                result = self.check_best_eval_higher_better(self.eval_metrics[idx],
                                                            eval_measures[idx], 
                                                            self.best_eval_measures_higher_better[idx-self.split_num],
                                                            self.best_eval_steps[idx], 
                                                            global_step
                                                            )
                if result != None:
                    self.best_eval_measures_higher_better[idx-self.split_num] = result['best_eval_measures_higher_better']
                    model_save_name = result['model_save_name']
                    self.best_eval_steps[idx] = result['best_eval_steps']   
                                                
                                                
            if result != None and self.is_checkpoint_save is True:
                if opt.distributed:
                    checkpoint = {'global_step': global_step,
                                    'model': model.module.state_dict(),
                                    'best_eval_measures_higher_better': self.best_eval_measures_higher_better,
                                    'best_eval_measures_lower_better': self.best_eval_measures_lower_better,
                                    'best_eval_steps': self.best_eval_steps
                    }
                else:
                    checkpoint = {'global_step': global_step,
                                    'model': model.state_dict(),
                                    'best_eval_measures_higher_better': self.best_eval_measures_higher_better,
                                    'best_eval_measures_lower_better': self.best_eval_measures_lower_better,
                                    'best_eval_steps': self.best_eval_steps
                    }
                torch.save(checkpoint, self.checkpoint_dir + model_save_name)
                print("Sucess to save '{}'.".format(model_save_name))
        
        if self.is_triple_train is True:
            result_commpute = {'eval_measures':loss_list, 
                            'val_sample': self.val_image_sample,
                            'val_image_tag_list': ['eh_val_origin', 'eh_val_second_est', 'eh_val_gt',
                                                    'de_val_est', 'de_val_gt', 'de_val_8x8', 'de_val_4x4', 'de_val_2x2',
                                                    'uncer_val_est2', 'uncer_err_val_gt', 'de_val_8x8_gt', 'de_val_4x4_gt', 'de_val_2x2_gt',],
                            'eval_metrics':self.eval_metrics, 
                            'error_string': error_string
                            }
        
        elif self.is_triple_train is False:
            result_commpute = {'eval_measures':loss_list, 
                            'val_sample': self.val_image_sample,
                            'val_image_tag_list': ['eh_val_origin', 'eh_val_second_est', 'eh_val_gt',
                                                    'de_val_est', 'de_val_gt', 'de_val_8x8', 'de_val_4x4', 'de_val_2x2',
                                                    'de_val_8x8_gt', 'de_val_4x4_gt', 'de_val_2x2_gt',],
                            'eval_metrics':self.eval_metrics, 
                            'error_string': error_string
                            }
        
        return result_commpute