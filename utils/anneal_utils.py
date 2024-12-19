import torch
import numpy as np

from utils.image_utils import psnr

from scene import GaussianModel

@torch.no_grad()
def cacl(gaussians, renderFunc, cams, renderArgs, type="psnr"):
    avg_psnr = 0.0
    for cam in cams:
        image = renderFunc(cam, gaussians, *renderArgs)["render"]

        _mask = None
        gt_image = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        avg_psnr += psnr(image, gt_image, _mask).mean().double()

    return avg_psnr / len(cams)

@torch.no_grad()
def transfer_gaussians(source_gaussians : GaussianModel, target_gaussians : GaussianModel, pick_rate):
    transfer_point_num = int(source_gaussians.get_xyz.shape[0] * pick_rate)
    vert_id_to_keep = np.random.choice(np.arange(source_gaussians.get_xyz.shape[0]), size=transfer_point_num, replace=False)
    vert_id_to_keep = torch.tensor(vert_id_to_keep).cuda().long()
    selected_pts_mask = torch.zeros((source_gaussians.get_xyz.shape[0])).cuda().bool()
    selected_pts_mask[vert_id_to_keep] = True

    new_xyz = source_gaussians._xyz[selected_pts_mask].clone()
    new_features_dc = source_gaussians._features_dc[selected_pts_mask].clone()
    new_features_rest = source_gaussians._features_rest[selected_pts_mask].clone()
    new_opacities = source_gaussians._opacity[selected_pts_mask].clone()
    new_scaling = source_gaussians._scaling[selected_pts_mask].clone()
    new_rotation = source_gaussians._rotation[selected_pts_mask].clone()
    new_origin_xyz = source_gaussians.origin_xyz[selected_pts_mask].clone()

    target_gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_origin_xyz)

    mask = torch.zeros((target_gaussians.get_xyz.shape[0])).cuda().bool()
    mask[-transfer_point_num:] = True

    return mask

@torch.no_grad()
def drop_gaussians(gaussians : GaussianModel, pick_rate):
    drop_point_num = int(gaussians.get_xyz.shape[0] * pick_rate)
    vert_id_to_keep = np.random.choice(np.arange(gaussians.get_xyz.shape[0]), size=drop_point_num, replace=False)
    vert_id_to_keep = torch.tensor(vert_id_to_keep).cuda().long()
    selected_pts_mask = torch.zeros((gaussians.get_xyz.shape[0])).cuda().bool()
    selected_pts_mask[vert_id_to_keep] = True

    drop_xyz = gaussians._xyz[selected_pts_mask].clone()
    drop_features_dc = gaussians._features_dc[selected_pts_mask].clone()
    drop_features_rest = gaussians._features_rest[selected_pts_mask].clone()
    drop_opacities = gaussians._opacity[selected_pts_mask].clone()
    drop_scaling = gaussians._scaling[selected_pts_mask].clone()
    drop_rotation = gaussians._rotation[selected_pts_mask].clone()
    drop_origin_xyz = gaussians.origin_xyz[selected_pts_mask].clone()

    gaussians.prune_points(selected_pts_mask, 0)

    return (drop_xyz, drop_features_dc, drop_features_rest, drop_opacities, drop_scaling, drop_rotation, drop_origin_xyz)
