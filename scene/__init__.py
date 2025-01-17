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

import os
import random
import json
import numpy as np
# import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_mvs import GaussianModelMVS
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360
from scene.cameras import PseudoCamera
# from gaussian_renderer import render
from utils.flow_utils import compute_depth_by_flow
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, gaussians_mvs : GaussianModelMVS, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        self.eval_cameras = {}
        self.train_dual_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, dataset_type=args.dataset_type)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.point_cloud)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Eval Cameras")
            self.eval_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.eval_cameras, resolution_scale, args)

            self.train_dual_cameras[resolution_scale] = []
            for i, cam_s in enumerate(self.train_cameras[resolution_scale]):
                for j, cam_t in enumerate(self.train_cameras[resolution_scale]):
                    if i == j:
                        continue
                    self.train_dual_cameras[resolution_scale].append((i, cam_s, j, cam_t))

            # pseudo_cams = []
            # if args.source_path.find('llff'):
            #     pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            # elif args.source_path.find('360'):
            #     pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            # view = self.train_cameras[resolution_scale][0]
            # for pose in pseudo_poses:
            #     pseudo_cams.append(PseudoCamera(
            #         R=pose[:3, :3].T, T=pose[:3, 3], K=view.K, FoVx=view.FoVx, FoVy=view.FoVy,
            #         width=view.image_width, height=view.image_height
            #     ))
            # self.pseudo_cameras[resolution_scale] = pseudo_cams

        #mvs_depths = gaussians_mvs._z.detach().view(gaussians_mvs.num, gaussians_mvs.H, gaussians_mvs.W)
        # for id, cam in enumerate(self.train_cameras[1.0]):
        #     render_pkg = render(cam, gaussians_mvs, pipe, background)
        #     self.train_cameras[1.0][id].set_mvs_depth(render_pkg["depth"])

        self.point_cloud = scene_info.point_cloud

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            for resolution_scale in resolution_scales:
                compute_depth_by_flow(self.train_cameras[resolution_scale], args.valid_dis_threshold, args.near_n, args.flow_type)
            self.gaussians.create_from_pcd(scene_info.point_cloud, gaussians_mvs.get_xyz, gaussians_mvs.get_features, None, self.cameras_extent, args.drop_rate)
            # self.gaussians.create_from_sparse_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def save_sparse(self, iteration, gaussians_sparse):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        gaussians_sparse.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTrainDualCameras(self, scale=1.0):
        return self.train_dual_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getEvalCameras(self, scale=1.0):
        return self.eval_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]
    
    def setGaussiansSparse(self, gaussians_sparse):
        self.gaussians_sparse = gaussians_sparse
        

class SceneMVS:

    gaussians : GaussianModelMVS

    def __init__(self, args : ModelParams, gaussians : GaussianModelMVS, plane_depths, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, dataset_type=args.dataset_type)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.point_cloud)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            # for cam in self.train_cameras[resolution_scale]:
            #     print(cam.image_name, cam.flow)
            # print(len(self.train_cameras[resolution_scale]), self.train_cameras[resolution_scale], "---------------------------------------------------------------------")

            # pseudo_cams = []
            # if args.source_path.find('llff'):
            #     pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            # elif args.source_path.find('360'):
            #     pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            # view = self.train_cameras[resolution_scale][0]
            # for pose in pseudo_poses:
            #     pseudo_cams.append(PseudoCamera(
            #         R=pose[:3, :3].T, T=pose[:3, 3], K=view.K, FoVx=view.FoVx, FoVy=view.FoVy,
            #         width=view.image_width, height=view.image_height
            #     ))
            # self.pseudo_cameras[resolution_scale] = pseudo_cams
        
        # cams = self.getTrainCameras().copy()
        # tgt_ids = []
        # cams_T = []
        # for i,cam in enumerate(cams):
        #     cams_T.append(cam.T)
        # cams_T = np.stack(cams_T)
        # mean_T = cams_T.mean(0)
        # ref_id = 0
        # min_dis = np.inf
        # for j,tgt_cam in enumerate(cams):
        #     tgt_t = tgt_cam.T
        #     dis = np.sum(np.square(mean_T - tgt_t))
        #     if dis < min_dis:
        #         dis = min_dis
        #         ref_id = j
        # self.ref_cam = cams[ref_id]

        # znear = self.ref_cam.znear
        # zfar = self.ref_cam.zfar
        # self.plane_depths = 1./torch.linspace(1./znear, 1./zfar, self.num_plane)
        self.plane_depths = plane_depths
        self.num_plane = plane_depths.shape[0]

        for resolution_scale in resolution_scales:
            compute_depth_by_flow(self.train_cameras[resolution_scale], args.valid_dis_threshold, args.near_n, args.flow_type)

        if args.dataset_type == 'dtu':

            for cam in self.train_cameras[1.0]:
                gt_image = cam.original_image.cuda()
                flow_depth = torch.tensor(cam.flow_depth).cuda()
                flow_depth_mask = torch.tensor(cam.flow_depth_mask).cuda().squeeze().unsqueeze(0)
                if 'scan110' not in args.source_path :
                    bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
                else:
                    bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)

                bg_mask_clone = bg_mask.clone()
                for i in range(1, 50):
                    bg_mask[:, i:] *= bg_mask_clone[:, :-i]
                flow_depth_mask[bg_mask.repeat(1,1,1)] = False

                cam.set_flow_depth(flow_depth, flow_depth_mask)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.plane_depths, self.train_cameras[1.0], self.cameras_extent, args.source_path)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_mvs/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]