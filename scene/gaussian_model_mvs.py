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
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, chamfer_dist
import open3d as o3d
from torch.optim.lr_scheduler import MultiStepLR

import cv2

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class GaussianModelMVS:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self.init_point = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.bg_color = torch.empty(0)
        self.confidence = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)
    
    @property
    def get_z(self):
        return (self.plane_z[self.masks] + self._dz)
    
    @property
    def get_mpi_z(self):
        return self.plane_mask * self.get_z.view(self.num, 1, self.H, self.W).repeat(self.num_plane, 1, 1, 1)
    
    # @property
    # def get_scaling(self):
    #     scale = torch.log(self.get_z.detach() / self.K[1,1]).repeat(1, 3)
    #     scale[:, 2] = math.log(0.0001)
    #     return self.scaling_activation(scale)
    
    @property
    def get_scaling(self):
        scale = torch.log(self.get_z.detach() / self.K[1,1]).repeat(1, 2)
        scale = torch.cat([scale, self._scaling], dim=1)
        return self.scaling_activation(scale)
    
    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        w = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    # @property
    # def get_xyz(self):
    #     xyz = (self.coords_ndc * self.get_z).view(-1, 1, 4)
    #     xyz = torch.bmm(xyz, self.full_proj_transform_invs)
    #     return xyz.view(-1,4)[:, :3]

    # @property
    # def get_xyz(self):
    #     xyz = (self.coords[self.masks] * self.get_z).view(-1, 1, 3)
    #     xyz = torch.bmm(xyz, self.K_invs[self.masks])
    #     xyz = torch.bmm(xyz - self.Ts[self.masks], self.R_invs[self.masks])
    #     return xyz.view(-1, 3)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, plane_depths, cams, spatial_lr_scale: float, source_path):
        num_plane = plane_depths.shape[0]
        self.num_plane = num_plane
        self.plane_depths = plane_depths
        zs = []
        colors = []
        full_proj_transform_invs = []
        Rs = []
        Ts = []
        pcd_idx = []
        pcd_origin_points = []
        masks = []
        coords = []

        num, H, W = len(cams), cams[0].image_height, cams[0].image_width
        self.num, self.H, self. W = num, H, W
        
        #pcd_points = torch.tensor(np.asarray(pcd.points)).cuda().float().view(-1, 3).repeat(num, 1)
        points_num = 0
        for i,cam in enumerate(cams):
            R = cam.R
            T = cam.T
            K = cam.K

            # points_views = np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)
            # cam_coord = np.matmul(K, points_views) ### for coordinate definition, see getWorld2View2() function
            # valid_idx = np.where(np.logical_and.reduce((cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=W-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=H-1)))[0]
            # cam_coord = cam_coord.transpose()
            # pts_depths = cam_coord[valid_idx, -1:]
            # cam_coord = cam_coord[valid_idx, :2] / cam_coord[valid_idx, -1:]
            # pcd_id = np.round(cam_coord[:, 1]).astype(np.int32) * W + np.round(cam_coord[:, 0]).astype(np.int32) + i * pcd.points.shape[0]
            # pcd_id = torch.tensor(pcd_id, dtype=torch.long).cuda()
            # pcd_idx.append(pcd_id)
            # pcd_origin_points.append(pcd_points[pcd_id])


            depth_reg = torch.tensor(cam.flow_depth).squeeze()
            flow_depth_mask = torch.tensor(cam.flow_depth_mask).squeeze()
            gt_image = cam.original_image.cuda().permute(1,2,0).contiguous()
            # full_proj_transform_inv = cam.full_proj_transform.inverse()
            full_proj_transform_inv = cam.full_proj_transform
            R = torch.tensor(R).cuda()
            T = torch.tensor(T).cuda()

            if cam.alpha_mask is not None:
                alpha_mask = cam.alpha_mask.cuda().squeeze().bool()
                flow_depth_mask = torch.logical_and(flow_depth_mask, alpha_mask)
            
            vaild_num = int(flow_depth_mask.sum())
            mask = flow_depth_mask.view(-1)
            
            zs.append(depth_reg.view(-1, 1)[mask])
            colors.append(gt_image.view(-1, 3)[mask])
            full_proj_transform_invs.append(full_proj_transform_inv.view(1,4,4).repeat(vaild_num, 1, 1))
            self.K = torch.tensor(cam.K).cuda().float()
            Rs.append(R.inverse().view(1,3,3).repeat(vaild_num, 1, 1))
            Ts.append(T.view(1,1,3).repeat(vaild_num, 1, 1))
            masks.append(flow_depth_mask.view(-1, 1)[mask])
            points_num += int(flow_depth_mask.sum())

            coords_y = torch.arange(0, H)
            coords_x = torch.arange(0, W)
            y, x = torch.meshgrid(coords_y, coords_x)
            y, x = y.unsqueeze(2), x.unsqueeze(2)
            coord = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().view(H*W, 3)[mask]
            coords.append(coord)
        
        # self.pcd_idx = torch.cat(pcd_idx, dim=0)
        # self.pcd_origin_points = torch.cat(pcd_origin_points, dim=0)
        zs = torch.cat(zs, dim=0).view(-1, 1).float().cuda()
        colors = torch.cat(colors, dim=0).cuda()
        self.colors = colors
        Rs = torch.cat(Rs, dim=0).cuda()
        Ts = torch.cat(Ts, dim=0).cuda()
        full_proj_transform_invs = torch.cat(full_proj_transform_invs, dim=0).cuda()
        self.masks = torch.cat(masks, dim=0).squeeze().cuda()

        coords_y = torch.arange(0, H).cuda()
        coords_x = torch.arange(0, W).cuda()
        # #coords_x = coords_x * (l - 1) / (height - 1)
        # #coords_y = coords_y * (l - 1) / (width - 1)
        y, x = torch.meshgrid(coords_y, coords_x)
        y, x = y.unsqueeze(2), x.unsqueeze(2)
        # x_ndc = (x * 2 + 1) / W - 1
        # y_ndc = (y * 2 + 1) / H - 1
        # self.coords_ndc = torch.stack([x_ndc, y_ndc, torch.ones_like(x), torch.ones_like(x)], dim=-1).float().view(H*W, 4).repeat(num, 1).view(-1, 4).float()
        # self.coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().view(H*W, 3).repeat(num, 1).view(-1, 3).float()
        self.coords = torch.cat(coords, dim=0).cuda().float()
        # self.full_proj_transform_invs = full_proj_transform_invs.view(num, 1, 1, 4, 4).repeat(1, H, W, 1, 1).view(-1, 4, 4).float()
        # self.R_invs = Rs.view(num, 1, 1, 3, 3).repeat(1, H, W, 1, 1).view(num * H * W, 3, 3).float()
        # self.Ts = Ts.view(num, 1, 1, 1, 3).repeat(1, H, W, 1, 1).view(num * H * W, 1, 3).float()
        self.full_proj_transform_invs = full_proj_transform_invs.float()
        self.R_invs = Rs.float()
        self.Ts = Ts.float()
        self.K_invs = self.K.inverse().view(1, 3, 3).repeat(points_num, 1, 1).float()

        # plane_depths_1 = plane_depths.view(1, -1).repeat(num * H * W, 1)
        # _, idx = torch.topk((plane_depths_1 - zs.repeat(1, num_plane)) ** 2, k=1, dim=1)
        # plane_mask = []
        # for i in range(num_plane):
        #     id_mask = torch.where(i == idx, 1., 0.).view(-1)
        #     plane_mask.append(id_mask)
        # self.plane_mask = torch.stack(plane_mask, dim=0).view(num_plane * num, 1, H, W)
        # self.plane_z = plane_depths[idx].view(-1, 1)
        # dz = zs - self.plane_z


        self.spatial_lr_scale = spatial_lr_scale
        #fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(colors).view(-1, 3)

        features = torch.zeros((points_num, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", points_num)

        xyz = (self.coords * zs).view(-1, 1, 3)
        xyz = torch.bmm(xyz, self.K_invs)
        xyz = torch.bmm(xyz - self.Ts, self.R_invs).view(points_num, 3)
        # dist2 = torch.clamp_min(distCUDA2(xyz)[0], 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.log(zs / self.K[1,1])
        #scales = torch.log(zs * self.K[1,1] / H).repeat(1, 3)
        #dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        #scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # dist2 = torch.ones((num * H * W)).float().cuda() * 0.01
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((points_num, 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((points_num, 1), dtype=torch.float, device="cuda"))


        self.pcd = pcd

        # self._dz = nn.Parameter(dz[self.masks].requires_grad_(True))
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0], self.get_xyz.shape[1], self.get_xyz.shape[2]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))
        
        # self.cacl_near()

        # path = os.path.join('/mnt/lab/zyl/models/FlowGS-final/dataset/mip360-colmap', os.path.split(source_path)[-1], '24_views/dense/ours')

        # if not os.path.exists(path):
        #     os.mkdir(path)

        # storePly(os.path.join(path, 'fused.ply'), self.get_xyz.detach().cpu().numpy(), colors[self.masks].detach().cpu().numpy() * 255.)

    def create_from_pcd1(self, pcd, plane_depths, cams, spatial_lr_scale: float, source_path):
        num_plane = plane_depths.shape[0]
        self.num_plane = num_plane
        self.plane_depths = plane_depths
        zs = []
        colors = []
        full_proj_transform_invs = []
        Rs = []
        Ts = []
        pcd_idx = []
        pcd_origin_points = []
        masks = []

        num, H, W = len(cams), cams[0].image_height, cams[0].image_width
        self.num, self.H, self. W = num, H, W
        
        #pcd_points = torch.tensor(np.asarray(pcd.points)).cuda().float().view(-1, 3).repeat(num, 1)
        points_num = 0
        for i,cam in enumerate(cams):
            R = cam.R
            T = cam.T
            K = cam.K

            # points_views = np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)
            # cam_coord = np.matmul(K, points_views) ### for coordinate definition, see getWorld2View2() function
            # valid_idx = np.where(np.logical_and.reduce((cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=W-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=H-1)))[0]
            # cam_coord = cam_coord.transpose()
            # pts_depths = cam_coord[valid_idx, -1:]
            # cam_coord = cam_coord[valid_idx, :2] / cam_coord[valid_idx, -1:]
            # pcd_id = np.round(cam_coord[:, 1]).astype(np.int32) * W + np.round(cam_coord[:, 0]).astype(np.int32) + i * pcd.points.shape[0]
            # pcd_id = torch.tensor(pcd_id, dtype=torch.long).cuda()
            # pcd_idx.append(pcd_id)
            # pcd_origin_points.append(pcd_points[pcd_id])


            depth_reg = torch.tensor(cam.flow_depth).cuda().squeeze()
            flow_depth_mask = torch.tensor(cam.flow_depth_mask).cuda().squeeze()
            gt_image = cam.original_image.cuda().permute(1,2,0).contiguous()
            # full_proj_transform_inv = cam.full_proj_transform.inverse()
            full_proj_transform_inv = cam.full_proj_transform
            R = torch.tensor(R).cuda()
            T = torch.tensor(T).cuda()

            if cam.alpha_mask is not None:
                alpha_mask = cam.alpha_mask.cuda().squeeze().bool()
                flow_depth_mask = torch.logical_and(flow_depth_mask, alpha_mask)
            
            zs.append(depth_reg.view(-1, 1))
            colors.append(gt_image.view(-1, 3))
            full_proj_transform_invs.append(full_proj_transform_inv.view(1,4,4).repeat(H * W, 1, 1))
            self.K = torch.tensor(cam.K).cuda().float()
            Rs.append(R.inverse().view(1,3,3).repeat(H * W, 1, 1))
            Ts.append(T.view(1,1,3).repeat(H * W, 1, 1))
            masks.append(flow_depth_mask.view(-1, 1))
            points_num += int(flow_depth_mask.sum())
        
        # self.pcd_idx = torch.cat(pcd_idx, dim=0)
        # self.pcd_origin_points = torch.cat(pcd_origin_points, dim=0)
        zs = torch.cat(zs, dim=0).view(-1, 1).float()
        colors = torch.cat(colors, dim=0)
        self.colors = colors
        Rs = torch.cat(Rs, dim=0)
        Ts = torch.cat(Ts, dim=0)
        full_proj_transform_invs = torch.cat(full_proj_transform_invs, dim=0)
        self.masks = torch.cat(masks, dim=0).squeeze()

        coords_y = torch.arange(0, H).cuda()
        coords_x = torch.arange(0, W).cuda()
        #coords_x = coords_x * (l - 1) / (height - 1)
        #coords_y = coords_y * (l - 1) / (width - 1)
        y, x = torch.meshgrid(coords_y, coords_x)
        y, x = y.unsqueeze(2), x.unsqueeze(2)
        x_ndc = (x * 2 + 1) / W - 1
        y_ndc = (y * 2 + 1) / H - 1
        self.coords_ndc = torch.stack([x_ndc, y_ndc, torch.ones_like(x), torch.ones_like(x)], dim=-1).float().view(H*W, 4).repeat(num, 1).view(-1, 4).float()
        self.coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().view(H*W, 3).repeat(num, 1).view(-1, 3).float()
        # self.full_proj_transform_invs = full_proj_transform_invs.view(num, 1, 1, 4, 4).repeat(1, H, W, 1, 1).view(-1, 4, 4).float()
        # self.R_invs = Rs.view(num, 1, 1, 3, 3).repeat(1, H, W, 1, 1).view(num * H * W, 3, 3).float()
        # self.Ts = Ts.view(num, 1, 1, 1, 3).repeat(1, H, W, 1, 1).view(num * H * W, 1, 3).float()
        self.full_proj_transform_invs = full_proj_transform_invs.float()
        self.R_invs = Rs.float()
        self.Ts = Ts.float()
        self.K_invs = self.K.inverse().view(1, 1, 1, 3, 3).repeat(num, H, W, 1, 1).view(num * H * W, 3, 3).float()

        plane_depths_1 = plane_depths.view(1, -1).repeat(num * H * W, 1)
        _, idx = torch.topk((plane_depths_1 - zs.repeat(1, num_plane)) ** 2, k=1, dim=1)
        plane_mask = []
        for i in range(num_plane):
            id_mask = torch.where(i == idx, 1., 0.).view(-1)
            plane_mask.append(id_mask)
        self.plane_mask = torch.stack(plane_mask, dim=0).view(num_plane * num, 1, H, W)
        self.plane_z = plane_depths[idx].view(-1, 1)
        dz = zs - self.plane_z


        self.spatial_lr_scale = spatial_lr_scale
        #fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(colors).view(-1, 3)

        features = torch.zeros((num * H * W, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", points_num)

        xyz = (self.coords * zs).view(-1, 1, 3)
        xyz = torch.bmm(xyz, self.K_invs)
        xyz = torch.bmm(xyz - self.Ts, self.R_invs)
        # dist2 = torch.clamp_min(distCUDA2(xyz)[0], 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.log(zs / self.K[1,1])
        #scales = torch.log(zs * self.K[1,1] / H).repeat(1, 3)
        #dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        #scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # dist2 = torch.ones((num * H * W)).float().cuda() * 0.01
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((num * H * W, 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((num * H * W, 1), dtype=torch.float, device="cuda"))


        self.pcd = pcd

        self._dz = nn.Parameter(dz[self.masks].requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1][self.masks].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:][self.masks].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales[self.masks].requires_grad_(True))
        self._rotation = nn.Parameter(rots[self.masks].requires_grad_(True))
        self._opacity = nn.Parameter(opacities[self.masks].requires_grad_(True))
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0], self.get_xyz.shape[1], self.get_xyz.shape[2]), device="cuda")
        self.confidence = torch.ones_like(opacities[self.masks], device="cuda")
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))
        
        # self.cacl_near()

        # path = os.path.join('/mnt/lab/zyl/models/FlowGS-final/dataset/mip360-colmap', os.path.split(source_path)[-1], '24_views/dense/ours')

        # if not os.path.exists(path):
        #     os.mkdir(path)

        # storePly(os.path.join(path, 'fused.ply'), self.get_xyz.detach().cpu().numpy(), colors[self.masks].detach().cpu().numpy() * 255.)


    def cacl_near(self):
        pcd = self.pcd
        xyz = (self.coords * self.get_z).view(-1, 1, 3)
        xyz = torch.bmm(xyz, self.K_invs)
        xyz = torch.bmm(xyz - self.Ts, self.R_invs)
        xyz_num = xyz.shape[0]
        pcd_num = pcd.shape[0]

        xyz_ori = []
        for idx,p in enumerate(xyz):
            p = p.view(1, 3).repeat(pcd_num, 1)
            dis = (p - pcd) ** 2
            dis = dis.sum(dim=-1)
            dis, sel = torch.topk(-dis, k=3, dim=0)
            sel_points = pcd[sel]
            xyz_ori.append(sel_points.mean(dim=0))

            if idx % 10000 == 0:
                print(idx)

        self.xyz_ori = torch.stack(xyz_ori, dim=0).cuda()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': 0.000016 * self.spatial_lr_scale, "name": "xyz"},
            #{'params': [self._z], 'lr': 0.0016 * self.spatial_lr_scale, "name": "xyz"},
            #{'params': [self._z], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.args.train_bg:
            l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = self.xyz_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = xyz_lr
                return xyz_lr
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        #for i in range(self._scaling.shape[1]):
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = torch.log(self.get_z / self.K[1,1]).repeat(1, 2)
        scale = torch.cat([scale, self._scaling], dim=1)
        #scale[:, 2] = math.log(0.0001)
        scale = scale.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
