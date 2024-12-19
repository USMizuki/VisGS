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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from gaussian_renderer import render, network_gui, render_point_mask
import sys
from scene import Scene, GaussianModel
from scene import SceneMVS, GaussianModelMVS
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
import torch.nn.functional as F
from torchvision import transforms

from utils.sh_utils import eval_sh

#from diffusers import StableDiffusionInpaintPipeline


def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask

def norm_mse_loss_global(input, target, margin, return_mask=False):
    input_patches = normalize(input.view(1, -1), std = input.std().detach())
    target_patches = normalize(target.view(1, -1), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def totalVariation(images, mask=None):
  if mask is None:
      mask = torch.ones_like(images).cuda()
  pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
  pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
  mask_dif1 = mask[:, :, 1:, :] * mask[:, :, :-1, :]
  mask_dif2 = mask[:, :, :, 1:] * mask[:, :, :, :-1]
  sum_axis = [1, 2, 3]

  tot_var = (
      torch.sum(torch.abs(pixel_dif1 * mask_dif1), dim=sum_axis) +
      torch.sum(torch.abs(pixel_dif2 * mask_dif2), dim=sum_axis))

  return tot_var / (images.shape[2]-1) / (images.shape[3]-1)


@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts, 0)


def gaussian_kernel(size: int, sigma: float):
    """生成一个 size x size 的高斯核"""
    x = torch.arange(size).float() - (size - 1) / 2
    gaussian = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = gaussian / gaussian.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

# 应用低通滤波器 (高斯模糊)
def apply_lowpass_filter(image_tensor, kernel_size=19, sigma=1.5):
    channels = image_tensor.size(1)
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).cuda()
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)  # 适应每个通道
    
    # 对每个通道应用高斯滤波
    filtered_image = F.conv2d(image_tensor, kernel, padding=kernel_size//2, groups=channels)
    return filtered_image

def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    viewpoint_stack, pseudo_stack = None, None

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_mvs = GaussianModelMVS(args)
    
    znear = 0.01
    zfar = 100
    plane_depths = 1./torch.linspace(1./znear, 1./zfar, 16).cuda()
    scene_mvs = SceneMVS(args, gaussians_mvs, plane_depths=plane_depths, shuffle=False)
    gaussians_mvs.training_setup(opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    progress_bar_mvs = tqdm(range(first_iter, opt.iterations_pre), desc="Pretraining progress")
    ema_loss_for_log = 0.0
    from utils.general_utils import vis_depth
    import cv2
    import utils_io
    # for id, cam in tqdm(enumerate(scene_mvs.train_cameras[1.0])):
    #     flow_depth = torch.tensor(cam.flow_depth).cuda()
    #     depth_map = vis_depth(flow_depth.detach().cpu().numpy())
    #     # cv2.imwrite(os.path.join(dataset.source_path, cam.image_name + '_depth' + '.png'), depth_map)
    #     np.save(os.path.join(dataset.source_path, 'ours_depth_maps', cam.image_name + '_depth.npy'), flow_depth.detach().cpu().numpy())
    #     utils_io.write_depth(os.path.join(dataset.source_path, 'ours_depth_maps', cam.image_name + '_depth'), flow_depth.detach().cpu().numpy(), bits=2)

    for iteration in range(1, opt.iterations_pre + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene_mvs.getTrainCameras().copy()
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians_mvs, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians_mvs, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

        # gt_image = viewpoint_cam.original_image.cuda()
        # mask = (gt_image!=0).sum(0, keepdim=True).bool().float()
        # Ll1 =  l1_loss_mask(image, gt_image, mask)
        # loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask)))

        depth = rendered_depth = render_pkg["depth"][0]
        flow_depth = torch.tensor(viewpoint_cam.flow_depth).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        flow_depth = flow_depth.reshape(-1, 1)


        #depth_loss = 1 - pearson_corrcoef( flow_depth, rendered_depth)
        #depth_loss = l1_loss_mask(flow_depth, rendered_depth)
        #loss += opt.depth_weight * depth_loss

        # tv_loss = totalVariation(gaussians_mvs._z.view(gaussians_mvs.num, 1, gaussians_mvs.H, gaussians_mvs.W))
        # loss += tv_loss.mean()

        lambda_tv = iteration / opt.iterations_pre
        # mpi_tv_loss = totalVariation(gaussians_mvs.get_z.view(gaussians_mvs.num, 1, gaussians_mvs.H, gaussians_mvs.W).repeat(gaussians_mvs.num_plane, 1, 1, 1), gaussians_mvs.plane_mask)
        # loss += mpi_tv_loss.mean() * lambda_tv

        depth_tv_loss = totalVariation(depth[None, None])
        #loss += depth_tv_loss.mean() * (1. - lambda_tv)

        #mask = gaussians_mvs.get_scaling.max(dim=1).values > 0.5
        #loss += gaussians_mvs.get_scaling[mask].max(dim=1).values.mean() * 1000

        
        loss.backward()
        gaussians_mvs.optimizer.step()
        gaussians_mvs.optimizer.zero_grad(set_to_none = True)

        with torch.no_grad():
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss,
            #                testing_iterations, scene_mvs, render, (pipe, background))
            if not loss.isnan():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar_mvs.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar_mvs.update(10)
            if iteration == opt.iterations_pre:
                progress_bar_mvs.close()

            if iteration in [3000, 5000, 10000, 20000, 30000]:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene_mvs.save(iteration)

            if iteration % 300000 == 0:
                gaussians_mvs.cacl_near()
            
            if iteration % 1000000 == 0:
                height, width = image.shape[1], image.shape[2]


                depth_norm = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                depth_mono = torch.tensor(viewpoint_cam.depth_image).cuda().unsqueeze(0)
                depth_mono_norm = (depth_mono - depth_mono.min()) / (depth_mono.max() - depth_mono.min())
                depth_reg = torch.tensor(viewpoint_cam.depth_reg).cuda().unsqueeze(0)
                depth_reg_norm = (depth_reg - depth_reg.min()) / (depth_reg.max() - depth_reg.min())

                tb_writer.add_images("a_images_src/ground_truth", gt_image[None], global_step=iteration)
                tb_writer.add_images("a_images_src/rendered", torch.clamp(image[None], 0., 1.0), global_step=iteration)
                tb_writer.add_images("a_images_src/depth", depth_norm[None, None], global_step=iteration)
                tb_writer.add_images("a_images_src/depth_mono", depth_mono_norm[None], global_step=iteration)
                tb_writer.add_images("a_images_src/depth_reg", depth_reg_norm[None], global_step=iteration)

    gaussians = GaussianModel(args)
    gaussians_sparse = GaussianModel(args)
    scene = Scene(args, gaussians, gaussians_mvs, shuffle=False)
    gaussians_sparse.create_from_sparse_pcd(scene.point_cloud, scene.cameras_extent)
    with torch.no_grad():
        if opt.iterations_pre > 0:
            from utils.general_utils import vis_depth
            import cv2
            for id, cam in tqdm(enumerate(scene.train_cameras[1.0])):
                render_pkg = render_point_mask(cam, gaussians_mvs, pipe, background, opt.prune_depth_scale)
                #render_pkg = render(cam, gaussians_mvs, pipe, background)
                scene.train_cameras[1.0][id].set_flow_depth(render_pkg["depth"], None)
        #if opt.diffusion_inpaint_iter != -1:
        # for id, cam in tqdm(enumerate(scene.eval_cameras[1.0])):
        #     render_pkg = render(cam, gaussians_mvs, pipe, background)
        #     image = render_pkg["render"]
        #     mask = (render_pkg["depth"] != 0.).float()
        #     scene.eval_cameras[1.0][id].set_mask(mask)

        #     torchvision.utils.save_image(image, os.path.join('output/inpaint', str(id) + '_rendered_image' + '.png'))
        #     torchvision.utils.save_image(image * mask, os.path.join('output/inpaint', str(id) + '_masked_image' + '.png'))
        #     torchvision.utils.save_image(mask, os.path.join('output/inpaint', str(id) + '_mask' + '.png'))
            

    gaussians.training_setup(opt)
    gaussians_sparse.training_setup_sparse(opt)
    scene.setGaussiansSparse(gaussians_sparse)

    del gaussians_mvs
    del scene_mvs
    torch.cuda.empty_cache()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    ema_loss_for_log = 0.0
    first_iter += 1
    viewpoint_stack = None
    viewpoint_stack_eval = None
    viewpoint_stack_pseudo = None

    # if args.dataset_type == 'blender':
    #     for cam in scene.getTrainCameras().copy():
    #         gaussians.prune(args, viewpoint_cam, None, opt.prune_depth_threshold)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
            gaussians_sparse.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        if not viewpoint_stack_eval:
            viewpoint_stack_eval = scene.getEvalCameras().copy()
        
        if not viewpoint_stack_pseudo:
            viewpoint_stack_pseudo = scene.getPseudoCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        viewpoint_cam_eval = viewpoint_stack_pseudo.pop(randint(0, len(viewpoint_stack_pseudo)-1))
        # viewpoint_cam_eval = viewpoint_stack_eval.pop(randint(0, len(viewpoint_stack_eval)-1))

        if args.dataset_type == 'dtu':
            if 'scan110' not in args.source_path :
                bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)

            #bg_mask = torch.logical_or(bg_mask, gt_image.min(0, keepdim=True).values > 254/255)
            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            gt_image[bg_mask.repeat(3,1,1)] = 0.

            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            (render_pkg["alpha"][bg_mask]**2).mean().backward()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
        elif args.dataset_type == 'blender':
            bg_mask = (gt_image.min(0, keepdim=True).values > 254/255)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        render_pkg_sparse = render(viewpoint_cam, gaussians_sparse, pipe, background)
        image_sparse, viewspace_point_tensor_sparse, visibility_filter_sparse, radii_sparse = render_pkg_sparse["render"], render_pkg_sparse["viewspace_points"], render_pkg_sparse["visibility_filter"], render_pkg_sparse["radii"]

        # Loss
        if args.dataset_type == 'dtu':
            # if 'scan110' not in args.source_path :
            #     bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
            # else:
            #     bg_mask = (gt_image.max(0, keepdim=True).values < 15/255)

            # #bg_mask = torch.logical_or(bg_mask, gt_image.min(0, keepdim=True).values > 254/255)
            # bg_mask_clone = bg_mask.clone()
            # for i in range(1, 50):
            #     bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            # gt_image[bg_mask.repeat(3,1,1)] = 1.
            Ll1 =  l1_loss_mask(image, gt_image)
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

            # mask = ((gt_image > 245./255.).sum(0) == 3).squeeze()
            # mask = torch.logical_or(mask, ((gt_image < 10./255.).sum(0) == 3).squeeze())
            # mask = ~mask
            # Ll1 =  l1_loss_mask(image, gt_image, mask.view(1, image.shape[1], image.shape[2]).float())
            # loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask.view(1, image.shape[1], image.shape[2]).float())))

        else:
            Ll1 =  l1_loss_mask(image, gt_image)
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

        Ll1_sparse =  l1_loss_mask(image_sparse, gt_image)
        loss_sparse = ((1.0 - opt.lambda_dssim) * Ll1_sparse + opt.lambda_dssim * (1.0 - ssim(image_sparse, gt_image)))


        # gt_image = viewpoint_cam.original_image.cuda()
        # mask = (gt_image!=0).sum(0, keepdim=True).bool().float()
        # Ll1 =  l1_loss_mask(image, gt_image, mask)
        # loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask)))

        rendered_depth = render_pkg["depth"]
        flow_depth = viewpoint_cam.flow_depth.cuda().unsqueeze(0)


        # if rendered_depth.shape[0] != 0 and iteration > 0 and opt.depth_weight != 0:
        #     if args.dataset_type == 'dtu':
        #         flow_depth[bg_mask] = flow_depth[~bg_mask].mean()
        #         rendered_depth[bg_mask] = rendered_depth[~bg_mask].mean().detach()
        #     elif args.dataset_type == 'blender':
        #         flow_depth[bg_mask] = 0

        #     flow_depth = flow_depth.view(-1,1)
        #     rendered_depth = rendered_depth.view(-1,1)
        #     #depth_loss = 1 - pearson_corrcoef( flow_depth, rendered_depth)

        #     # flow_depth_mask = 1 - viewpoint_cam.flow_depth_mask.cuda().float().view(-1,1)
        #     # depth_loss = l1_loss_mask(flow_depth, rendered_depth, flow_depth_mask)
        #     depth_loss = l1_loss_mask(flow_depth, rendered_depth)
        #     loss += opt.depth_weight * depth_loss
        
        # loss_sparse += 0.01 * l1_loss_mask(rendered_depth.detach(), render_pkg_sparse["depth"])

        render_pkg_sparse_eval = render(viewpoint_cam_eval, gaussians_sparse, pipe, background)
        image_sparse_eval = render_pkg_sparse_eval["render"]
        render_pkg_eval = render(viewpoint_cam_eval, gaussians, pipe, background)
        image_eval = render_pkg_eval["render"].detach()

        # image_sparse_eval = apply_lowpass_filter(image_sparse_eval.unsqueeze(0))
        # image_eval = apply_lowpass_filter(image_eval.unsqueeze(0)).squeeze()
        loss_sparse += 1.0 * ((1.0 - opt.lambda_dssim) * l1_loss_mask(image_eval, image_sparse_eval) + opt.lambda_dssim * (1.0 - ssim(image_sparse_eval, image_eval)))
        # loss_sparse += 0.01 * l1_loss_mask(render_pkg_sparse_eval["depth"], render_pkg_eval["depth"].detach())

        # if args.dataset_type == 'blender':
        #     loss_reg = torch.tensor(0., device=loss.device)
        #     shape_pena = (gaussians.get_scaling.max(dim=1).values / gaussians.get_scaling.min(dim=1).values).mean()
        #     # scale_pena = (gaussians.get_scaling.max(dim=1).values).std()
        #     scale_pena = ((gaussians.get_scaling.max(dim=1, keepdim=True).values)**2).mean()
        #     opa_pena = 1 - (opacity[opacity > 0.2]**2).mean() + ((1 - opacity[opacity < 0.2])**2).mean()

        #     # loss_reg += 0.01*shape_pena + 0.001*scale_pena + 0.01*opa_pena
        #     loss_reg += opt.shape_pena*shape_pena + opt.scale_pena*scale_pena + opt.opa_pena*opa_pena
        #     if iteration > opt.densify_until_iter:
        #         loss_reg *= 0.1

        #     loss += loss_reg

        # if iteration > opt.diffusion_inpaint_iter and opt.diffusion_inpaint_iter != -1:
        #     if not viewpoint_stack_eval:
        #         viewpoint_stack_eval = scene.getEvalCameras().copy()
        #     viewpoint_cam_eval = viewpoint_stack_eval.pop(randint(0, len(viewpoint_stack_eval)-1))
        #     render_pkg = render(viewpoint_cam_eval, gaussians, pipe, background)
        #     image_eval= render_pkg["render"]

        #     # Loss
        #     image_inpainted = viewpoint_cam_eval.original_image.cuda()
        #     Ll1 =  l1_loss_mask(image_eval, image_inpainted, 1-bg_mask.float())
        #     loss_eval = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_eval, image_inpainted, 1-bg_mask.float())))

        #     loss += 0.1 * loss_eval


        loss.backward(retain_graph=True)
        loss_sparse.backward(retain_graph=True)
        with torch.no_grad():
            # Progress bar
            if not loss.isnan():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "P": f"{gaussians.get_xyz.shape[0]}", "P_s": f"{gaussians_sparse.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            clean_iterations = testing_iterations + [first_iter]
            
            if args.dataset_type == 'dtu' or args.dataset_type == 'blender':
                clean_views(iteration, clean_iterations, scene, gaussians, pipe, background)
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # scene.save(iteration)
                scene.save_sparse(iteration, gaussians_sparse)

            # if iteration > first_iter and (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration),
            #                scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter and iteration not in clean_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                gaussians_sparse.max_radii2D[visibility_filter_sparse] = torch.max(gaussians_sparse.max_radii2D[visibility_filter_sparse], radii_sparse[visibility_filter_sparse])
                gaussians_sparse.add_densification_stats(viewspace_point_tensor_sparse, visibility_filter_sparse)
                # if iteration == opt.diffusion_inpaint_iter and opt.diffusion_inpaint_iter != -1:
                #     diffusion = StableDiffusionInpaintPipeline.from_pretrained(
                #         "checkpoints/runwayml/stable-diffusion-inpainting",
                #         revision="fp16",
                #         torch_dtype=torch.float16,
                #     ).to("cuda")
                #     for idx, cam in enumerate(scene.getEvalCameras()):
                #         render_pkg = render(cam, gaussians, pipe, background)
                #         image = render_pkg["render"]
                #         # render_pkg = render_point_mask(cam, gaussians, pipe, background)
                #         # point_depth = render_pkg["depth"]
                #         # mask = (point_depth != 0.).float()
                #         mask = cam.mask
                        
                #         prompt = "green leaves"
                #         #prompt = "flowers"
                #         image_inpaint = diffusion(prompt=prompt, image=image * mask, mask_image=1. - mask).images[0]
                #         image_inpaint = image_inpaint.resize((image.shape[2], image.shape[1]))
                #         image_inpaint.save(os.path.join('output/inpaint', str(idx) + '_inpainted' + '.png'))
                        
                #         torchvision.utils.save_image(image, os.path.join('output/inpaint', str(idx) + '_rendered_image' + '.png'))
                #         torchvision.utils.save_image(image * mask, os.path.join('output/inpaint', str(idx) + '_masked_image' + '.png'))
                #         torchvision.utils.save_image(mask, os.path.join('output/inpaint', str(idx) + '_mask' + '.png'))

                #         image_inpaint = transforms.ToTensor()(image_inpaint).cuda()
                        
                #         image_final = image_inpaint * (1. - mask) + image * mask
                #         cam.set_image(image_final)

                #         torchvision.utils.save_image(image_final, os.path.join('output/inpaint', str(idx) + '_inpainted_final' + '.png'))
                #     del diffusion

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = opt.size_threshold
                    
                    if args.dataset_type == 'blender':
                        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                        dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                        color = torch.clamp_min(sh2rgb + 0.5, 0.0)
                        # color = render(viewpoint_cam, gaussians, pipe, background)["color"]
                        white_mask = color.min(-1, keepdim=True).values > 253/255
                        gaussians.xyz_gradient_accum[white_mask] = 0
                        # gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(torch.ones_like(gaussians._opacity[white_mask]) * 0.1)
                        gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(gaussians.opacity_activation(gaussians._opacity[white_mask]) * 0.1)


                    #size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration, opt.dis_prune, opt.split_num)
                    if iteration > 30000:
                        gaussians_sparse.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration, opt.dis_prune, opt.split_num)

                    if args.dataset_type == 'blender':
                        if 'ship' in args.source_path: 
                            gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.5, 0)
                        if 'hotdog' in args.source_path: 
                            gaussians.prune_points(gaussians.get_xyz[:,-1] < -0.2, 0)      

                if iteration > opt.densify_from_iter and iteration % opt.prune_interval == 0 and opt.prune_interval != -1:
                    # gaussians.prune(scene.getTrainCameras(), opt.prune_depth_threshold)
                    gaussians.prune(args, viewpoint_cam, rendered_depth, opt.prune_depth_threshold)
                    gaussians_sparse.prune(args, viewpoint_cam, rendered_depth, opt.prune_depth_threshold)
                
                if iteration > opt.densify_from_iter and iteration % 500 == 0 and iteration <= 5000:
                    # vert_id_to_keep = np.random.choice(np.arange(gaussians.get_xyz.shape[0]), size=int(gaussians.get_xyz.shape[0] * 0.01), replace=False)
                    # vert_id_to_keep = torch.tensor(vert_id_to_keep).cuda().long()
                    # selected_pts_mask = torch.zeros((gaussians.get_xyz.shape[0])).cuda().bool()
                    # selected_pts_mask[vert_id_to_keep] = True

                    # grads = gaussians.xyz_gradient_accum / gaussians.denom
                    # grads[grads.isnan()] = 0.0
                    # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) < scene.cameras_extent * 0.01, True, False)
                    # print(selected_pts_mask.sum())

                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    mask = torch.where(torch.norm(grads, dim=-1) < scene.cameras_extent * 0.005, True, False)
                    # vert_id_to_keep = np.random.choice(np.arange(int(mask.sum())), size=min(int(mask.sum()), 1000), replace=False)
                    vert_id_to_keep = np.random.choice(np.arange(int(mask.sum())), size=int(0.001 * int(mask.sum())), replace=False)
                    # vert_id_to_keep = np.random.choice(np.arange(int(mask.sum())), size=min(int(mask.sum()), int(gaussians_sparse.get_xyz.shape[0] * 0.01)), replace=False)
                    vert_id_to_keep = torch.tensor(vert_id_to_keep).cuda().long()

                    selected_pts_mask_1 = torch.zeros((int(mask.sum()))).cuda().bool()
                    selected_pts_mask_1[vert_id_to_keep] = True
                    
                    selected_pts_mask = torch.zeros((gaussians.get_xyz.shape[0])).cuda().bool()
                    selected_pts_mask[mask] = selected_pts_mask_1

                    new_xyz = gaussians._xyz[selected_pts_mask].clone()
                    new_features_dc = gaussians._features_dc[selected_pts_mask].clone()
                    new_features_rest = gaussians._features_rest[selected_pts_mask].clone()
                    new_opacities = gaussians._opacity[selected_pts_mask].clone()
                    new_scaling = gaussians._scaling[selected_pts_mask].clone()
                    new_rotation = gaussians._rotation[selected_pts_mask].clone()
                    new_origin_xyz = gaussians.origin_xyz[selected_pts_mask].clone()

                    gaussians_sparse.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_origin_xyz)

            if iteration % 5000000 == 0:
                height, width = image.shape[1], image.shape[2]

                override_color = torch.sqrt(((gaussians.get_xyz - gaussians.origin_xyz) * (gaussians.get_xyz - gaussians.origin_xyz)).mean(1,keepdim=True)).repeat(1, 3)
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=override_color)
                confidence_map = render_pkg["render"][:1, :, :]
                depth = render_pkg["depth"]

                depth_norm = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                confidence_map_norm = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min())
                depth_mono = torch.tensor(viewpoint_cam.depth_image).cuda().unsqueeze(0)
                depth_mono_norm = (depth_mono - depth_mono.min()) / (depth_mono.max() - depth_mono.min())
                depth_reg = torch.tensor(viewpoint_cam.depth_reg).cuda().unsqueeze(0)
                depth_reg_norm = (depth_reg - depth_reg.min()) / (depth_reg.max() - depth_reg.min())

                tb_writer.add_images("a_images_src/ground_truth", gt_image[None], global_step=iteration)
                tb_writer.add_images("a_images_src/rendered", torch.clamp(image[None], 0., 1.0), global_step=iteration)
                tb_writer.add_images("a_images_src/confidence_map", confidence_map_norm[None], global_step=iteration)
                tb_writer.add_images("a_images_src/depth", depth_norm[None], global_step=iteration)
                tb_writer.add_images("a_images_src/depth_mono", depth_mono_norm[None], global_step=iteration)
                tb_writer.add_images("a_images_src/depth_reg", depth_reg_norm[None], global_step=iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                gaussians_sparse.optimizer.step()
                gaussians_sparse.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            gaussians_sparse.update_learning_rate(iteration)
            # if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
            #         iteration > args.start_sample_pseudo:
            #     gaussians.reset_opacity()
            #     gaussians_sparse.reset_opacity()
            
            if iteration % opt.opacity_reset_interval == 0:
                gaussians.reset_opacity()
                gaussians_sparse.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                l1_test_sparse, psnr_test_sparse, ssim_test_sparse, lpips_test_sparse = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    render_results_sparse = renderFunc(viewpoint, scene.gaussians_sparse, *renderArgs)
                    image_sparse = torch.clamp(render_results_sparse["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth = render_results["depth"]
                    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    l1_test_sparse += l1_loss(image_sparse, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips

                    _psnr = psnr(image_sparse, gt_image, _mask).mean().double()
                    _ssim = ssim(image_sparse, gt_image, _mask).mean().double()
                    _lpips = lpips(image_sparse, gt_image, _mask, net_type='vgg')
                    psnr_test_sparse += _psnr
                    ssim_test_sparse += _ssim
                    lpips_test_sparse += _lpips

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                psnr_test_sparse /= len(config['cameras'])
                ssim_test_sparse /= len(config['cameras'])
                lpips_test_sparse /= len(config['cameras'])
                l1_test_sparse /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                print("\n[ITER {}] Evaluating Sparse {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test_sparse, psnr_test_sparse, ssim_test_sparse, lpips_test_sparse))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            #tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[50_00, 10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # name = str(args.dataset_type) + '_' + str(args.images) + '_' + str(args.flow_type) + '_' + str(args.flow_checkpoint) + '_scalelr' + str(args.scaling_lr) + '_depth' + str(args.depth_weight) + '_near' + str(args.near_n) + '_valid' + str(args.valid_dis_threshold) + '_drop' + str(args.drop_rate) + '_N' + str(args.split_num)
    
    # args.model_path = os.path.join(args.model_path, name, os.path.split(args.source_path)[-1], str(args.n_views) + '_views')
    # print(args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")