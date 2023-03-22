import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffComposition(nn.Module): # differenable paste input img onto bg image 

    def __init__(self, input_img_size, output_img_size, device):

        super(DiffComposition, self).__init__()
        self.device = device
        self.x = nn.Parameter(torch.zeros(1, device=self.device))
        self.y = nn.Parameter(torch.zeros(1, device=self.device))
        self.w = nn.Parameter(torch.zeros(1, device=self.device))
        self.h = nn.Parameter(torch.zeros(1, device=self.device))

        self.input_size = input_img_size # n * c * h * w
        self.output_size = output_img_size # n * c * h * w




    def forward(self, src_img, bg_img): # src_img with shape input_shape, bg_img with shape output_shape
        self.norm_x = torch.sigmoid(self.x) * self.output_size[3]
        self.norm_y = torch.sigmoid(self.y) * self.output_size[2]
        self.norm_w = torch.sigmoid(self.w) * self.output_size[3]
        self.norm_h = torch.sigmoid(self.h) * self.output_size[2]

        eps = 1e-8
        trans_param = torch.cat([self.output_size[3] / (self.norm_w + eps), self.norm_x * 0.0, (2 / self.output_size[3] * (self.output_size[3] / 2 - self.norm_x)) * self.output_size[3] / (self.norm_w + eps), 
                                 self.norm_y * 0.0, self.output_size[2] / (self.norm_h + eps), (2 / self.output_size[2] * (self.output_size[2] / 2 - self.norm_y)) * self.output_size[2] / (self.norm_h + eps)], -1)
        trans_param = trans_param.view(-1, 2, 3)
        # affine grid with param = trans_param
        grid = F.affine_grid(trans_param, self.output_size).to(self.device)

        src_mask = torch.ones(src_img.shape).to(self.device)
        src_img_transform = F.grid_sample(src_img, grid)
        src_mask_transform = F.grid_sample(src_mask, grid)
        src_mask_transform_complement = 1 - src_mask_transform

        return bg_img * src_mask_transform_complement + src_img_transform * src_mask_transform, torch.cat([self.norm_x - self.norm_w / 2, self.norm_y - self.norm_h / 2, self.norm_x + self.norm_w / 2, self.norm_y + self.norm_h / 2], -1).unsqueeze(0)



