


import torch
import torch.nn as nn
import os

from timm.models.layers import to_2tuple
from ..network_builder import STRUCTURE, ENCODER
from .blocks.lpg import LocalPlanarGuidance_Block



class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.gelu1 = nn.GELU()
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=1, stride=1)
        self.gelu2 = nn.GELU()
        self.conv3x3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = torch.nn.functional.interpolate(x, scale_factor=self.ratio, mode='bilinear')
        out = self.gelu1(self.conv1x1(up_x))
        out = self.gelu2(self.conv3x3(out))
        return out



class TRIDENT(nn.Module):
    def __init__(self, 
                 encoder_model_cfg,  
                 max_depth,
                 predicted_coef_num=11,
                 ):
        super(TRIDENT, self).__init__()
        
        self.encoder = ENCODER.build(encoder_model_cfg)
        self.encoder_output_chan = self.encoder.skip_layer_output_channel
        
        self.max_depth = max_depth
        self.predicted_coef_num = predicted_coef_num
        
        
        self.de_attention_list = []
        for idx in range(len(self.encoder_output_chan)):
            
            if idx < 3:    
                de_add_layer_name = "de_attention_{}x{}".format(2**(idx+1), 2**(idx+1))          # de_attention_2x2, de_attention_4x4 ....
                de_add_layer = nn.Sequential(
                    nn.Conv2d(in_channels=self.encoder_output_chan[idx]//2,
                            out_channels=self.encoder_output_chan[idx], kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.encoder_output_chan[idx],
                            out_channels=self.encoder_output_chan[idx], kernel_size=3, padding=1, bias=False, groups=self.encoder_output_chan[idx]),
                    nn.BatchNorm2d(num_features=self.encoder_output_chan[idx]),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.encoder_output_chan[idx],
                            out_channels=self.encoder_output_chan[idx]//2, kernel_size=1)
                )

            else:    
                de_add_layer_name = "de_attention_{}x{}".format(2**(idx+1), 2**(idx+1))          # de_attention_2x2, de_attention_4x4 ....
                de_add_layer = nn.Sequential(
                    nn.Conv2d(in_channels=self.encoder_output_chan[idx]//2,
                            out_channels=self.encoder_output_chan[idx]//4, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.encoder_output_chan[idx]//4,
                            out_channels=self.encoder_output_chan[idx]//4, kernel_size=3, padding=1, bias=False, groups=self.encoder_output_chan[idx]//4),
                    nn.BatchNorm2d(num_features=self.encoder_output_chan[idx]//4),
                    nn.GELU(),
                    nn.Conv2d(in_channels=self.encoder_output_chan[idx]//4, out_channels=4, kernel_size=1)
                )
            
            self.add_module(de_add_layer_name, de_add_layer)
            self.de_attention_list.append(de_add_layer_name)  
        
        
        self.eh_attention_list = []
        for idx in range(len(self.encoder_output_chan)):
            eh_add_layer_name = "eh_attention_{}x{}".format(2**(idx+1), 2**(idx+1))          # de_attention_2x2, de_attention_4x4 ....
            eh_add_layer = nn.Sequential(
                nn.Conv2d(in_channels=self.encoder_output_chan[idx]//2,
                          out_channels=self.encoder_output_chan[idx], kernel_size=1),
                nn.GELU(),
                nn.Conv2d(in_channels=self.encoder_output_chan[idx],
                          out_channels=self.encoder_output_chan[idx], kernel_size=3, padding=1, bias=False, groups=self.encoder_output_chan[idx]),
                nn.BatchNorm2d(num_features=self.encoder_output_chan[idx]),
                nn.GELU(),
                nn.Conv2d(in_channels=self.encoder_output_chan[idx],
                          out_channels=self.encoder_output_chan[idx]//2, kernel_size=1)
            )
                
            self.add_module(eh_add_layer_name, eh_add_layer)
            self.eh_attention_list.append(eh_add_layer_name)   
        
     

        self.first_predicted_in_channels = self.encoder_output_chan[0]//2+self.encoder_output_chan[1]//2+self.encoder_output_chan[2]//2+self.encoder_output_chan[3]//2
        
        
        ######## for depth estimation & 2st predicted distortion coefficient  ########
        # Depth Estimation [16, 24, 40, 112]              
        self.pixelshuffle = nn.PixelShuffle(2)
        self.sigmoid16x16 = nn.Sigmoid()
        self.de_conv16x16 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[2]//2, 
                                               out_channels=self.encoder_output_chan[3]//8, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(num_features=self.encoder_output_chan[3]//8),
                                          nn.GELU())

        self.conv16x16_1 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[3]//8 + self.encoder_output_chan[3]//2 + 1, 
                                                   out_channels=self.encoder_output_chan[2]//2 + self.encoder_output_chan[3]//8, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.GELU())
        self.conv16x16_2 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[3]//8 + self.encoder_output_chan[3]//2 + 1, 
                                                   out_channels=self.encoder_output_chan[2]//2 + self.encoder_output_chan[3]//8, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.GELU(),
                                         nn.Conv2d(in_channels=self.encoder_output_chan[2]//2 + self.encoder_output_chan[3]//8, 
                                                   out_channels=self.encoder_output_chan[2]//2 + self.encoder_output_chan[3]//8, kernel_size=1, stride=1, bias=True),
                                         nn.GELU())
        self.eh_conv16x16 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[3]//8, 
                                                    out_channels=self.encoder_output_chan[3]//8, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.Conv2d(in_channels=self.encoder_output_chan[3]//8, 
                                                    out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
                                         )
        
        
        
                                     
        self.lpg8x8_block = LocalPlanarGuidance_Block(in_channels=self.encoder_output_chan[2]//2, max_depth=self.max_depth,
                                                                       lpg_upratio=8, use_grn=False)     # ch: 20 -> 11 = 2conv
        self.upconv8x8 = upconv(in_channels=self.encoder_output_chan[2]//2, out_channels=self.encoder_output_chan[2]//4)
        self.bn8x8 = nn.BatchNorm2d(self.encoder_output_chan[2]//4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv8x8_1 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[2]//4 + self.encoder_output_chan[2]//2 + 1, 
                                               out_channels=self.encoder_output_chan[1]//2 + self.encoder_output_chan[2]//2, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.GELU())
        self.conv8x8_2 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[2]//4 + self.encoder_output_chan[2]//2 + 1, 
                                               out_channels=self.encoder_output_chan[1]//2 + self.encoder_output_chan[2]//2, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.GELU(),
                                     nn.Conv2d(in_channels=self.encoder_output_chan[1]//2 + self.encoder_output_chan[2]//2, 
                                               out_channels=self.encoder_output_chan[1]//2 + self.encoder_output_chan[2]//2, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.GELU())
        self.eh_conv8x8 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[2]//2, 
                                                    out_channels=self.encoder_output_chan[2]//2, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.Conv2d(in_channels=self.encoder_output_chan[2]//2, 
                                                    out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
                                         )

                               
                                                                                        
        self.lpg4x4_block = LocalPlanarGuidance_Block(in_channels=self.encoder_output_chan[1]//2, max_depth=self.max_depth,
                                                                       lpg_upratio=4, use_grn=False)     # ch: 12 -> 7 = 2conv
        self.upconv4x4 = upconv(in_channels=self.encoder_output_chan[1]//2, out_channels=self.encoder_output_chan[1]//4)
        self.bn4x4 = nn.BatchNorm2d(self.encoder_output_chan[1]//4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4x4_1 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[1]//4 + self.encoder_output_chan[1]//2 +  1, 
                                               out_channels=self.encoder_output_chan[0]//2 + self.encoder_output_chan[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.GELU())
        self.conv4x4_2 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[1]//4 + self.encoder_output_chan[1]//2 +  1, 
                                               out_channels=self.encoder_output_chan[0]//2 + self.encoder_output_chan[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.GELU(),
                                     nn.Conv2d(in_channels=self.encoder_output_chan[0]//2 + self.encoder_output_chan[1]//2, 
                                               out_channels=self.encoder_output_chan[0]//2 + self.encoder_output_chan[1]//2, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.GELU())
        self.eh_conv4x4 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[1]//2, 
                                                    out_channels=self.encoder_output_chan[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.Conv2d(in_channels=self.encoder_output_chan[1]//2, 
                                                    out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
                                        )
        
    
        self.lpg2x2_block = LocalPlanarGuidance_Block(in_channels=self.encoder_output_chan[0]//2, max_depth=self.max_depth,
                                                                       lpg_upratio=2, use_grn=False)     # ch: 8 -> 5 = 2conv
        self.upconv2x2 = upconv(in_channels=self.encoder_output_chan[0]//2, out_channels=self.encoder_output_chan[0]//4)

        self.conv2x2_1 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[0]//4 + self.encoder_output_chan[0]//2 + 1, 
                                                 out_channels=self.encoder_output_chan[0]//4 + self.encoder_output_chan[0]//2, kernel_size=1, stride=1, bias=False),
                                      )     
        self.conv2x2_2 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[0]//4 + self.encoder_output_chan[0]//2 + 1, 
                                               out_channels=self.encoder_output_chan[0]//4 + self.encoder_output_chan[0]//2, kernel_size=1, stride=1, bias=False),
                                     nn.GELU(),
                                     nn.Conv2d(in_channels=self.encoder_output_chan[0]//4 + self.encoder_output_chan[0]//2, 
                                               out_channels=self.encoder_output_chan[0]//4 + self.encoder_output_chan[0]//2, kernel_size=3, stride=1, padding=1, bias=True),
                                     nn.GELU()
                                     )           
        self.eh_conv2x2 = nn.Sequential(nn.Conv2d(in_channels=self.encoder_output_chan[0]//2, 
                                                    out_channels=self.encoder_output_chan[0]//2, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.Conv2d(in_channels=self.encoder_output_chan[0]//2, 
                                                    out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
                                        )
        

        self.depth_extract_block = nn.Sequential(
            nn.Conv2d(in_channels=4+self.encoder_output_chan[0]//4, out_channels=self.encoder_output_chan[1], kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=self.encoder_output_chan[1], out_channels=self.encoder_output_chan[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=self.encoder_output_chan[0], out_channels=self.encoder_output_chan[0]//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=self.encoder_output_chan[0]//2, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
    
        self.eh_first_predicted_forw_1_0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.eh_first_predicted_forw_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.eh_first_predicted_forw_3_0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.eh_first_predicted_forw_concat = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=3,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )


        self.eh_first_predicted_forw_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=1, stride=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=4, groups=4, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=1, stride=1, bias=False)
        )
        
        self.eh_first_predicted_forw_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=1, stride=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=4, groups=4, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=1, stride=1, bias=False)
        )
        
        self.eh_first_predicted_forw_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=1, stride=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=4, groups=4, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=1, stride=1, bias=False)
        )


        self.eh_first_predicted_backw_1_0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.eh_first_predicted_backw_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.eh_first_predicted_backw_3_0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.eh_first_predicted_backw_concat = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=3,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )
        
        self.eh_first_predicted_backw_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=1, stride=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=4, groups=4, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=1, stride=1, bias=False)
        )

        self.eh_first_predicted_backw_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=1, stride=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=4, groups=4, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=1, stride=1, bias=False)
        )
        
        self.eh_first_predicted_backw_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4,
                      kernel_size=1, stride=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=4, groups=4, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=1,
                      kernel_size=1, stride=1, bias=False)
        )

        self.eh_first_predicted_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=3,
                      kernel_size=1, stride=1, bias=False)
        )

        self.eh_first_predicted_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=3,
                      kernel_size=1, stride=1, bias=False)
        )


        self.sigmoid = nn.Sigmoid()

        
        
        
    

    def init_img_size_check(self, img_size):
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            pass
        else:
            raise TypeError("args: type'img_size' is must be 'int' or 'tuple', but Got {}".format(type(img_size)))     
        return img_size


    def checkpoint_loader(self, checkpoint_path, model, strict=True):
        space1 = "".rjust(5)
         
        if os.path.isfile(checkpoint_path):
            print(space1+"🚀 Start Loading checkpoint '{}'".format(checkpoint_path))
            
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'], strict=strict)
            
            print(space1+"🚀 Loaded checkpoint '{}'".format(checkpoint_path))
        else:
            print(space1+"🚀 No checkpoint found at '{}'".format(checkpoint_path))
            raise ValueError("🚀 No checkpoint found at '{}'".format(checkpoint_path))

        return model

    
    def forward(self, x):

        
        ########################### Encoder part ############################# 
        enhanced_x = self.encoder(x)     # len=4, 1/2,1/4,1/8,1/16        

        enhanced_attention = []
        depth_attention = []
        enhanced_input = []
        depth_input = []
        idx = 0
        for enhanced_layer_name, depth_layer_name in zip(self.eh_attention_list, self.de_attention_list):
            split_channel = self.encoder_output_chan[idx]//2
            
            enhanced_input.append(enhanced_x[idx][:,:split_channel,:,:])
            depth_input.append(enhanced_x[idx][:,:split_channel:,:,:])
            
            eh_data = self._modules[enhanced_layer_name](enhanced_input[idx])   
            de_data = self._modules[depth_layer_name](depth_input[idx])   
            enhanced_attention.append(eh_data)                      # enhanced attention: 2x2, 4x4, 8x8, 16x16
            depth_attention.append(de_data)
            idx = idx+1   

        
        ########################### Decoder part #############################         
        ######## for depth estimation ########        
        depth_scale_list = []
        enhance_scale_list = []

        # Depth & Uncertainty of Depth & Distortion Coefficients for 16x16
        depth_16x16_tmp = self.sigmoid16x16(self.pixelshuffle(depth_attention[3]))
        depth_16x16_scaled = torch.nn.functional.interpolate(depth_16x16_tmp, scale_factor=8, mode='bilinear')
        depth_scale_list.append(depth_16x16_scaled)
        enhance16x16_tmp = torch.nn.functional.interpolate(enhanced_attention[3], scale_factor=2, mode='bilinear')

        depth16x16_upconv = self.de_conv16x16(depth_attention[2])
        input_tmp = torch.concat([depth_16x16_tmp, depth16x16_upconv, enhance16x16_tmp], dim=1)
        input_tmp = self.conv16x16_1(input_tmp) + self.conv16x16_2(input_tmp)
        
        split_channel = self.encoder_output_chan[2]//2   # -> 8x8 depth channels
        depth_16x16_tmp = input_tmp[:,:split_channel,:,:]
        enhanced_16x16_second = input_tmp[:,split_channel:,:,:]
        enhance_scale_list.append(torch.nn.functional.interpolate(self.eh_conv16x16(enhanced_16x16_second), scale_factor=8, mode='bilinear'))



        # Depth & Uncertainty of Depth & Distortion Coefficients for 8x8
        depth8x8_scaled = self.lpg8x8_block(depth_16x16_tmp)
        depth_scale_list.append(depth8x8_scaled)

        depth_8x8_tmp = torch.nn.functional.interpolate(depth8x8_scaled, scale_factor=1/4, mode='bilinear')
        enhance8x8_tmp = torch.nn.functional.interpolate(enhanced_attention[2], scale_factor=2, mode='bilinear')

        depth8x8_upconv = self.bn8x8(self.upconv8x8(depth_attention[2]))
        input_tmp = torch.concat([depth_8x8_tmp, depth8x8_upconv, enhance8x8_tmp], dim=1)
        input_tmp = self.conv8x8_1(input_tmp) + self.conv8x8_2(input_tmp)
        
        split_channel = self.encoder_output_chan[1]//2   # -> 8x8 depth channels
        depth_8x8_tmp = input_tmp[:,:split_channel,:,:]
        enhanced_8x8_second = input_tmp[:,split_channel:,:,:]
        enhance_scale_list.append(torch.nn.functional.interpolate(self.eh_conv8x8(enhanced_8x8_second), scale_factor=4, mode='bilinear'))


        # Depth & Uncertainty of Depth & Distortion Coefficients for 4x4
        depth4x4_scaled = self.lpg4x4_block(depth_8x8_tmp)
        depth_scale_list.append(depth4x4_scaled)    

        depth_4x4_tmp = torch.nn.functional.interpolate(depth4x4_scaled, scale_factor=1/2, mode='bilinear')
        enhance4x4_tmp = torch.nn.functional.interpolate(enhanced_attention[1], scale_factor=2, mode='bilinear')
        
        depth4x4_upconv = self.bn4x4(self.upconv4x4(depth_attention[1]))
        input_tmp = torch.concat([depth_4x4_tmp, depth4x4_upconv, enhance4x4_tmp], dim=1)
        input_tmp = self.conv4x4_1(input_tmp) + self.conv4x4_2(input_tmp) 
        
        split_channel = self.encoder_output_chan[0]//2   # -> 4x4 depth channels
        depth_4x4_tmp = input_tmp[:,:split_channel,:,:]
        enhanced_4x4_second = input_tmp[:,split_channel:,:,:]
        enhance_scale_list.append(torch.nn.functional.interpolate(self.eh_conv4x4(enhanced_4x4_second), scale_factor=2, mode='nearest'))


        # Depth & Uncertainty of Depth & Distortion Coefficients for 2x2
        depth2x2_scaled = self.lpg2x2_block(depth_4x4_tmp)
        depth_scale_list.append(depth2x2_scaled)

        depth_2x2_tmp = depth2x2_scaled
        enhance2x2_tmp = torch.nn.functional.interpolate(enhanced_attention[0], scale_factor=2, mode='nearest')
        
        depth2x2_upconv = self.upconv2x2(depth_attention[0])
        input_tmp = torch.concat([depth_2x2_tmp, depth2x2_upconv, enhance2x2_tmp], dim=1)
        input_tmp = self.conv2x2_1(input_tmp) + self.conv2x2_2(input_tmp)


        split_channel = self.encoder_output_chan[0]//4   # -> 2x2 depth channels
        
        depth_input_tmp = input_tmp[:,:split_channel,:,:]
        enhanced_2x2_second = input_tmp[:,split_channel: ,:,:]
        enhance_scale_list.append(self.eh_conv2x2(enhanced_2x2_second))

        
        ######## for depth estimation & image enhancement ########
        depth_tmp = torch.concat([depth_16x16_scaled, depth8x8_scaled, depth4x4_scaled, depth2x2_scaled, depth_input_tmp], dim=1)
        depth_scaled = self.depth_extract_block(depth_tmp) 
        depth = depth_scaled * self.max_depth
        

        Trans_value_list = []
        first_forw_tmp_1 = torch.concat([enhance_scale_list[0][:,0,:,:].unsqueeze(1), 
                                         enhance_scale_list[1][:,0,:,:].unsqueeze(1), 
                                         enhance_scale_list[2][:,0,:,:].unsqueeze(1),
                                         enhance_scale_list[3][:,0,:,:].unsqueeze(1)], dim=1)
        first_forw_tmp_1 = self.eh_first_predicted_forw_1_0(first_forw_tmp_1)

        
        first_forw_tmp_2 = torch.concat([enhance_scale_list[0][:,1,:,:].unsqueeze(1), 
                                         enhance_scale_list[1][:,1,:,:].unsqueeze(1), 
                                         enhance_scale_list[2][:,1,:,:].unsqueeze(1),
                                         enhance_scale_list[3][:,1,:,:].unsqueeze(1)], dim=1)
        first_forw_tmp_2 = self.eh_first_predicted_forw_2_0(first_forw_tmp_2)
        

        
        first_forw_tmp_3 = torch.concat([enhance_scale_list[0][:,2,:,:].unsqueeze(1), 
                                         enhance_scale_list[1][:,2,:,:].unsqueeze(1), 
                                         enhance_scale_list[2][:,2,:,:].unsqueeze(1),
                                         enhance_scale_list[3][:,2,:,:].unsqueeze(1)], dim=1)
        first_forw_tmp_3 = self.eh_first_predicted_forw_3_0(first_forw_tmp_3)
        
        first_forw_concat = self.eh_first_predicted_forw_concat(torch.concat([first_forw_tmp_1, first_forw_tmp_2, first_forw_tmp_3], dim=1))
        first_forw_tmp_r = first_forw_concat[:,0,:,:].unsqueeze(1)
        first_forw_tmp_g = first_forw_concat[:,1,:,:].unsqueeze(1)
        first_forw_tmp_b = first_forw_concat[:,2,:,:].unsqueeze(1)
        
        Trans_value_list.append(first_forw_tmp_r)
        Trans_value_list.append(first_forw_tmp_g)
        Trans_value_list.append(first_forw_tmp_b)
        
        first_forw_tmp_1_1 = self.eh_first_predicted_forw_1_1(x[:,0,:,:].unsqueeze(1) * first_forw_tmp_r)
        first_forw_tmp_2_1 = self.eh_first_predicted_forw_2_1(x[:,1,:,:].unsqueeze(1) * first_forw_tmp_g)
        first_forw_tmp_3_1 = self.eh_first_predicted_forw_3_1(x[:,2,:,:].unsqueeze(1) * first_forw_tmp_b)
        
        first_forw_tmp = torch.concat([first_forw_tmp_1_1, first_forw_tmp_2_1, first_forw_tmp_3_1], dim=1)
                 
        
        BackScat_value_list = []
        first_backw_tmp_1 = torch.concat([enhance_scale_list[0][:,3,:,:].unsqueeze(1), 
                                          enhance_scale_list[1][:,3,:,:].unsqueeze(1), 
                                          enhance_scale_list[2][:,3,:,:].unsqueeze(1),
                                          enhance_scale_list[3][:,3,:,:].unsqueeze(1)], dim=1)
        
        first_backw_tmp_1 = self.eh_first_predicted_backw_1_0(first_backw_tmp_1)
        

                            
        first_backw_tmp_2 = torch.concat([enhance_scale_list[0][:,4,:,:].unsqueeze(1), 
                                          enhance_scale_list[1][:,4,:,:].unsqueeze(1), 
                                          enhance_scale_list[2][:,4,:,:].unsqueeze(1),
                                          enhance_scale_list[3][:,4,:,:].unsqueeze(1)], dim=1)
        first_backw_tmp_2 = self.eh_first_predicted_backw_2_0(first_backw_tmp_2)



        first_backw_tmp_3 = torch.concat([enhance_scale_list[0][:,5,:,:].unsqueeze(1), 
                                          enhance_scale_list[1][:,5,:,:].unsqueeze(1), 
                                          enhance_scale_list[2][:,5,:,:].unsqueeze(1),
                                          enhance_scale_list[3][:,5,:,:].unsqueeze(1)], dim=1)
        first_backw_tmp_3 = self.eh_first_predicted_backw_3_0(first_backw_tmp_3)
        
        
        first_backw_concat = self.eh_first_predicted_backw_concat(torch.concat([first_backw_tmp_1, first_backw_tmp_2, first_backw_tmp_3], dim=1))
        first_backw_tmp_r = first_backw_concat[:,0,:,:].unsqueeze(1)
        first_backw_tmp_g = first_backw_concat[:,1,:,:].unsqueeze(1)
        first_backw_tmp_b = first_backw_concat[:,2,:,:].unsqueeze(1)
        
        BackScat_value_list.append(first_backw_tmp_r)
        BackScat_value_list.append(first_backw_tmp_g)
        BackScat_value_list.append(first_backw_tmp_b)
        
        first_backw_tmp_1_1 = self.eh_first_predicted_backw_1_1(first_backw_tmp_r * first_forw_tmp_r)
        first_backw_tmp_2_1 = self.eh_first_predicted_backw_2_1(first_backw_tmp_g * first_forw_tmp_g)
        first_backw_tmp_3_1 = self.eh_first_predicted_backw_3_1(first_backw_tmp_b * first_forw_tmp_b)
        
        first_backw_tmp = torch.concat([first_backw_tmp_1_1, first_backw_tmp_2_1, first_backw_tmp_3_1], dim=1)
        
    
        first_forw_tmp = self.eh_first_predicted_1(first_forw_tmp)
        first_backw_tmp = self.eh_first_predicted_2(first_backw_tmp)
        
        first_predicted = first_forw_tmp + first_backw_tmp
        first_predicted = self.sigmoid(first_predicted)
        
        return depth, depth_scale_list, enhanced_x, first_predicted, Trans_value_list, BackScat_value_list
    


class Uncertainty_Network(nn.Module):
    def __init__(self, 
                 encoder_output_chan
                 ):
        
        super(Uncertainty_Network, self).__init__()
        self.encoder_output_chan = encoder_output_chan
 
        self.uncertainty_16x16_first = nn.Sequential(
            nn.Conv2d(in_channels=self.encoder_output_chan[3], out_channels=self.encoder_output_chan[3]//4,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=self.encoder_output_chan[3]//4, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

        
        self.uncertainty_8x8_first = nn.Sequential(
            nn.Conv2d(in_channels=self.encoder_output_chan[2], out_channels=self.encoder_output_chan[2]//2,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=self.encoder_output_chan[2]//2, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )
                
        
        self.uncertainty_4x4_first = nn.Sequential(
            nn.Conv2d(in_channels=self.encoder_output_chan[1], out_channels=self.encoder_output_chan[1]//2,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=self.encoder_output_chan[1]//2, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )  
        

        self.uncertainty_2x2_first = nn.Sequential(
            nn.Conv2d(in_channels=self.encoder_output_chan[0], out_channels=self.encoder_output_chan[0]//2,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=self.encoder_output_chan[0]//2, out_channels=6,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=2,
                      kernel_size=3, stride=1, padding=1, bias=False)
        ) 


        # case 0
        input_channels = 2 + 2 + 2 + 2
        
        self.uncertainty_extract_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=input_channels, out_channels=4, kernel_size=1)
        )
        
        forw_channels = 6
        
        self.uncertainty_extract_2 = nn.Sequential(
            nn.Conv2d(in_channels=forw_channels, out_channels=input_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=input_channels, out_channels=4, kernel_size=1)
        )
        

        self.uncertainty_extract_layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    
    def forward(self, x, encoder_input, Trans_value_list):
        
        encoder_feature_2x2, encoder_feature_4x4, encoder_feature_8x8, encoder_feature_16x16 = encoder_input

        feature_16x16 = self.uncertainty_16x16_first(encoder_feature_16x16.detach())
        feature_16x16 = torch.nn.functional.interpolate(feature_16x16, scale_factor=16, mode='bilinear')

        feature_8x8 = self.uncertainty_8x8_first(encoder_feature_8x8.detach())
        feature_8x8 = torch.nn.functional.interpolate(feature_8x8, scale_factor=8, mode='bilinear')
        
        feature_4x4 = self.uncertainty_4x4_first(encoder_feature_4x4.detach())
        feature_4x4 = torch.nn.functional.interpolate(feature_4x4, scale_factor=4, mode='bilinear')
        
        feature_2x2 = self.uncertainty_2x2_first(encoder_feature_2x2.detach())
        feature_2x2 = torch.nn.functional.interpolate(feature_2x2, scale_factor=2, mode='bilinear')
                        
        
        # case 0                      
        uncertainty_tmp_1 = torch.concat([feature_16x16, feature_8x8, feature_4x4, feature_2x2], dim=1)
        uncertainty_tmp_1 = self.uncertainty_extract_1(uncertainty_tmp_1) 
        
        t_r = Trans_value_list[0].detach()
        t_g = Trans_value_list[1].detach()
        t_b = Trans_value_list[2].detach()
        
        uncertainty_tmp_2 = self.uncertainty_extract_2(torch.concat([x, t_r, t_g, t_b], dim=1))
        uncertainty = self.uncertainty_extract_layer(uncertainty_tmp_1 + uncertainty_tmp_2)    
        
        return uncertainty

    
    
@STRUCTURE.register_module()
class TRIDENT_MODEL(nn.Module):
    def __init__(self, 
                 encoder_model_cfg,  
                 max_depth,
                 predicted_coef_num=11,
                 is_use_uncertainty = False,
                 ):
        super(TRIDENT_MODEL, self).__init__()

        self.joint_structure =  TRIDENT(encoder_model_cfg=encoder_model_cfg,
                                        max_depth=max_depth,
                                        predicted_coef_num=predicted_coef_num
                                        )
        
        
        self.is_use_uncertainty = is_use_uncertainty      # "multi_task" / "triple_task"
        self.encoder_output_chan = self.joint_structure.encoder_output_chan
        
        
        if is_use_uncertainty is True:
            self.uncertainty_network = Uncertainty_Network(encoder_output_chan=self.encoder_output_chan)
        self.max_depth = max_depth
        
 

    def forward(self, x):
        
        depth, depth_scale_list, enhanced_x, first_predicted, Trans_value_list, BackScat_value_list = self.joint_structure(x)
        
        if self.is_use_uncertainty: 
            uncertainty =  self.uncertainty_network(x, enhanced_x, Trans_value_list)
            uncertainty = uncertainty * self.max_depth
            return depth, depth_scale_list, uncertainty, first_predicted, Trans_value_list, BackScat_value_list
        
        else:
            return depth, depth_scale_list, first_predicted, Trans_value_list, BackScat_value_list
  