#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
import cv2
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, K, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device = "cuda", mask = None, bounds=None, flow=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K.transpose()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        if mask is not None:
            self.alpha_mask = torch.tensor(mask)
        else:
            self.alpha_mask = None
        self.bounds = bounds
        self.flow = flow

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def set_flow_depth(self, flow_depth, flow_depth_mask):
        self.flow_depth = flow_depth.clone().squeeze().cpu()
        if flow_depth_mask is not None:
            self.flow_depth_mask = flow_depth_mask.clone().squeeze().cpu()
        # blur_depth = cv2.blur(self.flow_depth, (9,9), )
        # self.blur_flow_depth = np.where(self.flow_depth < blur_depth, self.flow_depth, blur_depth)
    
    def set_image(self, image):
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
    
    def set_mask(self, mask):
        self.mask = mask.to(self.data_device)




class PseudoCamera(nn.Module):
    def __init__(self, R, T, K, FoVx, FoVy, width, height, trans=np.array([0.0, 0.0, 0.0]), scale=1.0 ):
        super(PseudoCamera, self).__init__()

        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def set_image(self, image):
        self.original_image = image.cpu()

    def set_image_and_mask(self, image):
        mask = torch.where(image == 0, 1, 0)
        mask = mask[:1] * mask[1:2] * mask[2:]
        mask = mask ^ 1
        self.pseudo_image = image.cpu().numpy()
        self.pseudo_image_mask = mask.cpu().numpy()
    
    def set_flow_depth(self, flow_depth, flow_depth_mask):
        self.flow_depth = flow_depth.cpu().numpy()
        if flow_depth_mask is not None:
            self.flow_depth_mask = flow_depth_mask.cpu().numpy()



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

