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
from utils.sh_utils import eval_sh
from gtracer import GaussianTracer
import trimesh


class GaussianModel:

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

    def __init__(self, args, transmittance_min=0.001):
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

        icosahedron = trimesh.creation.icosahedron()
        self.unit_icosahedron_vertices = torch.from_numpy(icosahedron.vertices).float().cuda() * 1.2584 
        self.unit_icosahedron_faces = torch.from_numpy(icosahedron.faces).long().cuda()

        self.gaussian_tracer = GaussianTracer(transmittance_min=transmittance_min)
        self.alpha_min = 1 / 255

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
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        w = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

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

    def create_from_pcd(self, pcd: BasicPointCloud, points, features, scaling, spatial_lr_scale: float, drop_rate):
        
        vert_id_to_keep = np.random.choice(np.arange(points.shape[0]), size=int(points.shape[0] * drop_rate), replace=False)
        vert_id_to_keep = torch.tensor(vert_id_to_keep).cuda().long()
        fused_point_cloud = points[vert_id_to_keep]
        features = features[vert_id_to_keep]
        features = features.transpose(1, 2)

        self.origin_xyz = fused_point_cloud.clone()

        # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # #fused_point_cloud = gaussians_mvs.get_xyz.view(-1, 3)
        # #fused_color = gaussians_mvs.colors.view(-1, 3)

        # features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # if self.args.use_color:
        #     features[:, :3, 0] =  fused_color
        # features[:, 3:, 1:] = 0.0

        self.spatial_lr_scale = spatial_lr_scale

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        #scales = scaling
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.live_count = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.int, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))
    
    def create_from_sparse_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        self.origin_xyz = fused_point_cloud.clone()
        #fused_point_cloud = gaussians_mvs.get_xyz.view(-1, 3)
        #fused_color = gaussians_mvs.colors.view(-1, 3)

        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.args.use_color:
            features[:, :3, 0] =  fused_color
        features[:, 3:, 1:] = 0.0

        self.spatial_lr_scale = spatial_lr_scale

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        #scales = scaling
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.live_count = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.int, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.ones_like(opacities, device="cuda")
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))




    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.args.train_bg:
            l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def training_setup_sparse(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0.05, "name": "scaling"},
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
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.05))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def dist_prune(self):
        dist = chamfer_dist(self.init_point, self._xyz)
        valid_points_mask = (dist < 3.0)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def prune_points(self, mask, iter):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.confidence = self.confidence[valid_points_mask]
        self.live_count = self.live_count[valid_points_mask]
        self.origin_xyz = self.origin_xyz[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['bg_color']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_origin_xyz):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)

        new_live_count = torch.zeros((new_xyz.shape[0], 1), dtype=torch.int, device="cuda")
        self.live_count = torch.cat((self.live_count, new_live_count), dim=0)
        self.origin_xyz = torch.cat((self.origin_xyz, new_origin_xyz), dim=0)


    def proximity(self, scene_extent, N = 3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)



    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        # dist, _ = distCUDA2(self.get_xyz)
        # selected_pts_mask2 = torch.logical_and(dist > (self.args.dist_thres * scene_extent),
        #                                        torch.max(self.get_scaling, dim=1).values > ( scene_extent))
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)

        stds = self.get_scaling[selected_pts_mask].repeat(N - 1, 1) * (N - 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N - 1, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N - 1, 1)
        new_xyz = torch.cat([new_xyz, self.get_xyz[selected_pts_mask]], dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_origin_xyz = self.origin_xyz[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_origin_xyz)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_origin_xyz = self.origin_xyz[selected_pts_mask]


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_origin_xyz)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter, dis_prune, split_num):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, iter, N=split_num)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # if max_screen_size:
        #     big_points_ws = self.get_scaling.max(dim=1).values > max_screen_size * extent
        #     prune_mask = big_points_ws
        if max_screen_size and max_screen_size != -1:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if dis_prune:
            dis = torch.sqrt(((self.origin_xyz - self.get_xyz.detach()) ** 2).mean(-1))
            dis_prune_mask = (dis > 2).squeeze()
            self.prune_points(dis_prune_mask, iter)
        else:
            self.prune_points(prune_mask, iter)

        torch.cuda.empty_cache()
    
    # def prune(self, cams, prune_depth_threshold):
    #     points = self.get_xyz
    #     mask = torch.zeros((points.shape[0]), dtype=torch.bool).cuda()
    #     for cam in cams:
    #         image = cam.original_image.cuda()
    #         H, W = cam.image_height, cam.image_width
    #         depth = torch.tensor(cam.blur_flow_depth).cuda().squeeze()
    #         #depth = torch.tensor(cam.flow_depth).cuda().squeeze()

    #         pro_xyz = torch.cat([points, torch.ones_like(points[:, :1]).cuda()], dim=1) @ cam.full_proj_transform
    #         z = pro_xyz[:, 3:]
    #         xy = pro_xyz[:, :2] / (z + 0.0000001)

    #         x = ((xy[:, :1] + 1.) * W - 1.) * 0.5
    #         y = ((xy[:, 1:] + 1.) * H - 1.) * 0.5
            
    #         x = torch.round(x).long().squeeze()
    #         y = torch.round(y).long().squeeze()
    #         z = z.squeeze()

    #         valid = torch.logical_and(x >= 0, x <= W-1)
    #         valid = torch.logical_and(valid, torch.logical_and(y >=0, y <= H-1)).squeeze()
    #         real_depth = depth[y[valid], x[valid]]
    #         real_rgb = image[:, y[valid], x[valid]].sum(0)

    #         #mask[valid] = torch.logical_or(mask[valid], z[valid] < real_depth - prune_depth_threshold)
    #         mask[valid] = torch.logical_or(mask[valid], real_rgb==0)

    #     self.prune_points(mask, 0)
    
    def prune(self, args, cam, rendered_depth, prune_depth_threshold):
        points = self.get_xyz
        mask = torch.zeros((points.shape[0]), dtype=torch.bool).cuda()
        # for cam in cams:
            
        #alpha_mask = torch.tensor(cam.alpha_mask).cuda()
        image = cam.original_image.cuda()
        H, W = cam.image_height, cam.image_width
        #depth = torch.tensor(cam.blur_flow_depth).cuda().squeeze()
        depth = torch.tensor(cam.flow_depth).cuda().squeeze()

        pro_xyz = torch.cat([points, torch.ones_like(points[:, :1]).cuda()], dim=1) @ cam.full_proj_transform
        z = pro_xyz[:, 3:]
        xy = pro_xyz[:, :2] / (z + 0.0000001)

        x = ((xy[:, :1] + 1.) * W - 1.) * 0.5
        y = ((xy[:, 1:] + 1.) * H - 1.) * 0.5
        
        x = torch.round(x).long().squeeze()
        y = torch.round(y).long().squeeze()
        z = z.squeeze()

        valid = torch.logical_and(x >= 0, x <= W-1)
        valid = torch.logical_and(valid, torch.logical_and(y >=0, y <= H-1)).squeeze()

        # depth = depth.view(H, W)
        # rendered_depth = rendered_depth.view(H, W)

        # depth = torch.min(depth, rendered_depth)
        # #cam.set_flow_depth(depth, None)
        # depth = depth[y[valid], x[valid]]

        if cam.alpha_mask is not None:
            alpha_mask = cam.alpha_mask.cuda()
            real_alpha = alpha_mask[y[valid], x[valid]]

            mask[valid] = torch.logical_or(mask[valid], real_alpha==0)

        # real_rgb = image[:, y[valid], x[valid]].transpose(0, 1)
        # mask[valid] = torch.logical_or(mask[valid], ((real_rgb > 254./255.).sum(1) == 3).squeeze())
        # mask[valid] = torch.logical_or(mask[valid], ((real_rgb < 1./255.).sum(1) == 3).squeeze())

        # shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        # dir_pp = (self.get_xyz - cam.camera_center.repeat(self.get_features.shape[0], 1))
        # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        # sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        # if args.dataset_type == 'dtu':
        #     if 'scan110' not in args.source_path :
        #             bg_mask = (colors_precomp.max(1).values < 30/255)
        #     else:
        #         bg_mask = (colors_precomp.max(1).values < 15/255)

        #     mask = torch.logical_or(mask, bg_mask)
        # # mask = torch.logical_or(mask, colors_precomp.min(1).values > 254/255)
        # if args.white_background:
        #     mask = torch.logical_or(mask, colors_precomp.min(1).values > 254/255)
        # else:
        #     mask = torch.logical_or(mask, colors_precomp.max(1).values < 1/255)

        # if prune_depth_threshold != -1:
        #     mask[valid] = torch.logical_or(mask[valid], z[valid] < depth - prune_depth_threshold)

        self.prune_points(mask, 0)
            


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def add_points(self, viewpoint_cam, image, depth, confidence_map, condidence_coeff):
        _, H, W = image.shape
        K = torch.tensor(viewpoint_cam.K).cuda()

        xs = torch.linspace(0, W - 1, W).float()
        ys = torch.linspace(0, H - 1, H).float()

        xs = xs.view(1, W, 1).repeat(H, 1, 1).cuda()
        ys = ys.view(H, 1, 1).repeat(1, W, 1).cuda()
        xyzs = torch.cat((xs, ys, torch.ones(xs.size()).float().cuda()), dim=2).view(-1, 3)

        image = image.permute(1,2,0).contiguous().view(-1, 3)
        depth = depth.view(-1, 1)
        confidence_map = confidence_map.permute(1,2,0).contiguous().view(-1, 1)

        confidence_mask = (confidence_map > condidence_coeff).squeeze()

        colors = image[confidence_mask]
        xyzs = xyzs[confidence_mask]
        depth = depth[confidence_mask]

        projected_coors = xyzs * depth
        xyz_source = projected_coors @ torch.inverse(K)
        xyz_source = torch.cat((xyz_source, torch.ones_like(xyz_source[:, 0:1])), dim=1)

        xyz_world = xyz_source @ torch.inverse(viewpoint_cam.world_view_transform)
        new_xyz = xyz_world[:, 0:3]

        fused_color = RGB2SH(colors)

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        new_features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        P = new_xyz.shape[0]
        dist, _ = distCUDA2(torch.cat((new_xyz, self.get_xyz), dim=0))
        dist2 = torch.clamp_min(dist, 0.0000001)[:P]
        new_scaling = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        new_rotation = torch.zeros((new_xyz.shape[0], 4), device="cuda")
        new_rotation[:, 0] = 1
        
        #new_features = torch.zeros((new_xyz.shape[0], 32)).float().cuda()
        new_opacities = inverse_sigmoid(0.1 * torch.ones((new_xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        #new_depth_err = torch.zeros_like(new_opacities)
        new_origin_xyz = new_xyz.clone()
        new_xyz_weight = torch.ones((new_xyz.shape[0], 1), dtype=torch.float, device="cuda") * self.alpha_init

        #self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features, new_opacities, new_scaling, new_rotation, new_depth_err, new_origin_xyz, new_xyz_weight)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_origin_xyz, new_xyz_weight)

    def get_boundings(self, alpha_min=0.01):
        mu = self.get_xyz
        opacity = self.get_opacity
        
        L = build_scaling_rotation(self.get_scaling, self._rotation)
        
        vertices_b = (2 * (opacity/alpha_min).log()).sqrt()[:, None] * (self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces_b = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device="cuda")[:, None, None] * 12
        gs_id = torch.arange(mu.shape[0], device="cuda")[:, None].expand(-1, faces_b.shape[1])
        return vertices_b.reshape(-1, 3), faces_b.reshape(-1, 3), gs_id.reshape(-1)

    def get_SinvR(self):
        return build_scaling_rotation(1 / self.get_scaling, self._rotation)
    
    def build_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.gaussian_tracer.build_bvh(vertices_b, faces_b, gs_id)
        
    def update_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.gaussian_tracer.update_bvh(vertices_b, faces_b, gs_id)
        
    def trace(self, rays_o, rays_d):
        SinvR = self.get_SinvR()
        means3D = self.get_xyz
        shs = self.get_features
        opacity = self.get_opacity
        colors, depth, alpha = self.gaussian_tracer.trace(rays_o, rays_d, means3D, opacity, SinvR, shs, alpha_min=self.alpha_min, deg=self.active_sh_degree)
        return {
            "render": colors,
            "depth": depth,
            "alpha" : alpha,
        }

    def prune_invisible(self, cams):
        rays_o = self.get_xyz
        prune_mask = torch.ones((rays_o.shape[0]), dtype=bool, device="cuda")
        for cam in cams:
            rays_d = cam.camera_center.view(1,3) - rays_o
            results = self.trace(rays_o, rays_d)
            prune_mask = torch.logical_and(prune_mask, results["alpha"] > 0.95)
        self.prune_points(prune_mask, 0)