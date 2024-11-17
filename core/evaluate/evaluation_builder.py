
from mmcv.utils import Registry

EVALUATOR = Registry('evaluator')
EVALUATOR_BUILDER = Registry('evaluator_builder')


@EVALUATOR_BUILDER.register_module()
class Build_Evaluator(object):
    def __init__(self, evaluator_cfg_list:list, device, dataloader_eval, save_dir, ngpus=None):
        super().__init__()
                 
        self.evaluator_list = []
        for evaluator_cfg in evaluator_cfg_list:
            evaluator_cfg['device'] = device
            evaluator_cfg['ngpus'] = ngpus
            evaluator_cfg['save_dir'] = save_dir
            evaluator_cfg['dataloader_eval'] = dataloader_eval
            
            self.evaluator_list.append(EVALUATOR.build(evaluator_cfg))    
        
    
    def result_evaluation(self, opt, model, global_step):
        final_commpute = []
        
        for evaluator in self.evaluator_list:
            final_commpute.append(evaluator.evalutate_worker(opt, model, global_step))
        
        return final_commpute
    