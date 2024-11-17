
# 
import torch.nn as nn
import torch
import time
import glob

eval_file_list = glob.glob('result_de.kitti.jointformer_testing/eval/*')
eval_file_list.sort()

# gt_file_list = glob.glob('result_de.kitti.jointformer_testing/eval_gt/*')
# gt_file_list.sort()

depth_scaling = '256'
print(len(eval_file_list))
# print(len(gt_file_list))

# assert len(eval_file_list) == len(gt_file_list)

with open('sample.txt', 'w') as f:
    for index in range(len(eval_file_list)):
        f.write(eval_file_list[index] + ' ')
        # f.write(gt_file_list[index] + ' ')
        # f.write(depth_scaling)
        f.write('\n')
f.close()