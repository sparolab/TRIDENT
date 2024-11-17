import torch
import torch.nn as nn

import math





                
class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                                 kernel_size=1, stride=1, padding=0),
                                                                       nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        
        return net




class Reduc_1x1_Block(nn.Module):
    def __init__(self, in_channels,  max_depth, use_grn=False):
        super().__init__()
        
        self.max_depth = max_depth
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        out_channels = in_channels

        
        while out_channels > 8:
            self.reduc.add_module('inter_{}2{}'.format(in_channels, out_channels),
                                    torch.nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                                bias=False, kernel_size=1, stride=1),
                                                        nn.GELU()))
            in_channels = out_channels
            out_channels = out_channels//2
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=3,
                      kernel_size=1,
                      bias=False
                      )
        )
        
    def forward(self, x):      
        net = self.reduc.forward(x)
                
        depth_theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
        depth_phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
        depth_dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
        depth_n1 = torch.mul(torch.sin(depth_theta), torch.cos(depth_phi)).unsqueeze(1)
        depth_n2 = torch.mul(torch.sin(depth_theta), torch.sin(depth_phi)).unsqueeze(1)
        depth_n3 = torch.cos(depth_theta).unsqueeze(1)
        depth_n4 = depth_dist.unsqueeze(1)
        
        depth_coef = torch.cat([depth_n1, depth_n2, depth_n3, depth_n4], dim=1)
        
        return depth_coef


class LocalPlanarGuidance(nn.Module):
    def __init__(self, upratio):
        super(LocalPlanarGuidance, self).__init__()
        
        self.upratio = upratio
        
        self.u = torch.arange(int(self.upratio)).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        
        self.upratio_f = float(upratio)
        
        
    def forward(self, plane_eq):
        
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio_f), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio_f), 3)
        
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]
        
        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio_f), plane_eq.size(3)).cuda()
        # u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio_f), plane_eq.size(3))
        u = (u - (self.upratio_f - 1) * 0.5) / self.upratio_f
        
        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio_f)).cuda()
        # v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio_f))
        v = (v - (self.upratio_f - 1) * 0.5) / self.upratio_f

        return n4 / (n1 * u + n2 * v + n3)


class LocalPlanarGuidance_Block(nn.Module):
    def __init__(self, in_channels, max_depth, lpg_upratio, use_grn=False):
        super().__init__()

        self.max_depth = max_depth
        self.lpg_upratio = lpg_upratio
        self.reduc1x1 = Reduc_1x1_Block(in_channels=in_channels,
                                        max_depth=self.max_depth,
                                        use_grn=use_grn)
        
        self.lpg_depth = LocalPlanarGuidance(upratio=lpg_upratio)
        
        
    def forward(self, x):

        depth_coef = self.reduc1x1(x)
        
        depth_plane_coef = depth_coef[:, :3, :, :]
        depth_plane_normal = torch.nn.functional.normalize(depth_plane_coef, 2, 1)
        depth_plane_dist = depth_coef[:, 3, :, :]
                
        depth_plane_eq = torch.cat([depth_plane_normal, depth_plane_dist.unsqueeze(1)], 1)

        depth = self.lpg_depth(depth_plane_eq)
        depth_scaled = depth.unsqueeze(1) / self.max_depth

        return depth_scaled


