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
from utils import utils_io

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
    output_path = os.path.join(args.source_path, 'ours_depth_maps')
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
    for id, cam in tqdm(enumerate(scene_mvs.train_cameras[1.0])):
        flow_depth = torch.tensor(cam.flow_depth).cuda()
        depth_map = vis_depth(flow_depth.detach().cpu().numpy())
        cv2.imwrite(os.path.join('output/inpaint', cam.image_name + '_depth' + '.png'), depth_map)

        filename = cam.image_name
        name = 'depth_'+filename.split('/')[-1]
        print('######### output_path and name:', output_path,  name)
        output_file_name = os.path.join(output_path, name.split('.')[0])
        # utils.io.write_depth(output_file_name.split('.')[0], output, bits=2)
        utils_io.write_depth(output_file_name, flow_depth.cpu().numpy(), bits=2)



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
                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth = render_results["depth"]
                    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
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