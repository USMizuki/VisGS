import torch
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

midas = torch.hub.load("/mnt/lab/zyl/.cache/torch/hub/intel-isl_MiDaS_master", "DPT_Hybrid", source='local')
#midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load("/mnt/lab/zyl/.cache/torch/hub/intel-isl_MiDaS_master", "transforms", source='local')
#midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
downsampling = 1

model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_NK", source="local", pretrained=True).to('cuda')

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            prediction = model_zoe.infer(img[None]).view(h ,w)
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        prediction = model_zoe.infer(img[None]).view(h ,w)
    return prediction



def regularize_depth(pcd, R, T, K, FoVx, FoVy, resolution, depth):
    width, height = resolution
    points_views = np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)
    cam_coord = np.matmul(K, points_views) ### for coordinate definition, see getWorld2View2() function
    valid_idx = np.where(np.logical_and.reduce((cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=width-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=height-1)))[0]
    pts_depths = cam_coord[-1:, valid_idx]
    cam_coord = cam_coord[:2, valid_idx] / cam_coord[-1:, valid_idx]
    cam_coord[0, :] /= (width - 1.)
    cam_coord[1, :] /= (height - 1.)
    cam_coord = torch.from_numpy(cam_coord).cuda().permute(1,0).float() * 2 -1.
    pts_depths = torch.from_numpy(pts_depths).cuda().float()
    sample_depths = torch.nn.functional.grid_sample(depth[None, None], cam_coord[None, None], align_corners=True).view(-1, 1)
    
    zfar = 100.0
    znear = 0.01
    world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    points = torch.from_numpy(pcd.points).float().cuda()
    points_view = torch.cat((points, torch.ones((points.shape[0], 1)).cuda()), dim=1) @ world_view_transform
    points_view = points_view @ projection_matrix
    pts_depths1 = points_view[:, -1:]
    points_view = points_view / (points_view[:, -1:] + 0.0000001)
    points_xy = points_view[:, :2]
    vaild_mask = torch.logical_and(points_xy[:, 0] >= -1., points_xy[:, 0] <= 1.)
    vaild_mask = torch.logical_and(vaild_mask, torch.logical_and(points_xy[:, 1] >= -1., points_xy[:, 1] <= 1.))
    points_xy = points_xy[vaild_mask]
    sample_depths1 = torch.nn.functional.grid_sample(depth[None, None], points_xy[None, None], align_corners=True).view(-1, 1)


    #A = torch.cat((1 / sample_depths, torch.ones_like(sample_depths).cuda()), dim=1)
    #B = pts_depths.squeeze()

    #X = torch.linalg.lstsq(A, B).solution
    #result = A @ X
    #dis = (A @ X - B).mean()
    #depth_reg = (1 / depth) * X[0] + X[1]

    scale = torch.ones(1).cuda().requires_grad_(True)
    scale1 = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)
    shift1 = (torch.ones(1) * 0.5).cuda().requires_grad_(True)
    weight = torch.ones_like(sample_depths1).squeeze().cuda().requires_grad_(True)

    #optimizer = torch.optim.Adam(params=[scale, shift, weight], lr=1.0)
    optimizer = torch.optim.Adam(params=[scale, scale1, shift, shift1], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0
    
    while abs(loss_ema - loss_prev) > 1e-5:
    # while loss > 1e-5:
        source_hat = scale*sample_depths1 + scale1 / (sample_depths1 + shift1)
        #loss = torch.mean((((pts_depths1 - source_hat).squeeze() * torch.softmax(weight, dim=0))**2)) + weight.std()
        loss = torch.mean((((pts_depths1 - source_hat).squeeze())**2))

        # penalize depths not in [0,1]
        loss_hinge1 = loss_hinge2 = 0.0
        if (source_hat<=0.0).any():
            loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
        # if (source_hat>=1.0).any():
        #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
        loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}, {scale1.item():.4f},{shift1.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    with torch.no_grad():
        depth_reg = (depth) * scale + scale1 / (depth + shift1)
    return depth_reg

