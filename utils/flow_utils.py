import numpy as np
import torch

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
        
def compute_depth_by_flow(cams, valid_dis_threshold, n, flow_type):
    for i, cam_s in enumerate(cams):
        H, W = cam_s.image_height, cam_s.image_width
        R_s = torch.tensor(cam_s.R.transpose()).cuda().float()
        T_s = torch.tensor(cam_s.T.transpose()).cuda().float().view(3, 1)
        K_s = torch.tensor(cam_s.K.transpose()).cuda().float()
        image_s = cam_s.original_image.cuda()


        cams_near = []
        for j, flow in cam_s.flow:
            cam_t = cams[j]
            R_t = torch.tensor(cam_t.R.transpose()).cuda().float()
            T_t = torch.tensor(cam_t.T.transpose()).cuda().float().view(3, 1)
            cam_dis = ((R_s.inverse() @ T_s - R_t.inverse() @ T_t) ** 2).sum(0)
            cam_dis = torch.sqrt(cam_dis)
            cams_near.append((j, float(cam_dis), flow))
        cams_near = sorted(cams_near, key=lambda x:x[1])
        # n = 3
        # for j, cam_dis, _ in cams_near:
        #     if cam_dis < 5:
        #         n = j + 1

        
        if flow_type not in ['avg', 'dis', 'grad', 'weight', 'disxgrad']:
            assert False, "Could not recognize flow type!"

        id_dis = 100000
        final_confidence = None
        final_depth = None
        final_mask = None

        if flow_type == 'weight':
            depths = []
            weights = []
        for j, _, flow in cams_near[:min(n, len(cams_near))]:
            # if abs(i - j) > 1:
            #     continue
            flow = torch.tensor(flow).cuda()
            cam_t = cams[j]
            image_t = cam_t.original_image.cuda()

            R_t = torch.tensor(cam_t.R.transpose()).cuda().float()
            T_t = torch.tensor(cam_t.T.transpose()).cuda().float().view(3, 1)
            K_t = torch.tensor(cam_t.K.transpose()).cuda().float()

            coords_y = torch.arange(0, H).cuda()
            coords_x = torch.arange(0, W).cuda()
            y, x = torch.meshgrid(coords_y, coords_x)
            y, x = y.unsqueeze(2), x.unsqueeze(2)
            xyz_s = torch.cat([x, y, torch.ones_like(x).cuda().float()], dim=0).view(3, -1)
            

            R = R_t @ R_s.inverse()
            T = R_s @ R_t.inverse() @ T_t - T_s
            T = T.squeeze()
            S = torch.zeros((3, 3)).cuda()
            S[0, 1] = - T[2]
            S[1, 0] = T[2]
            S[0, 2] = T[1]
            S[2, 0] = - T[1]
            S[1, 2] = - T[0]
            S[2, 1] = T[0]
            E = R @ S
            F = K_t.inverse().transpose(0, 1) @ E @ K_s.inverse()

            epipolar_line = F @ xyz_s
            a = epipolar_line[0:1]
            b = epipolar_line[1:2]
            c = epipolar_line[2:3]

            u = (x.contiguous().view(H, W, 1) + flow[:, :, :1]).contiguous().view(1, -1)
            v = (y.contiguous().view(H, W, 1) + flow[:, :, 1:2]).contiguous().view(1, -1)

            x_t_near = (b * b * u - a * b * v - a * c) / (a ** 2 + b ** 2)
            y_t_near = (a * a * v - a * b * u - b * c) / (a ** 2 + b ** 2)
            xyz_t_near = torch.cat([x_t_near, y_t_near, torch.ones_like(x_t_near).cuda()], dim=0)

            dis = torch.abs((a * u + b * v + c) / torch.sqrt(a ** 2 + b ** 2))
            dis_max = dis.max()
            mask = (dis < valid_dis_threshold)
            #mask = torch.ones_like(mask, dtype=torch.bool).cuda()
            # xyz_t = torch.cat([u, v, torch.ones_like(u).cuda()], dim=0)
            # mask_num = mask.sum()
            # test1 = ((F.transpose(0, 1) @ xyz_t_near) * xyz_s).sum(dim=0)
            # test2 = ((F.transpose(0, 1) @ xyz_t) * xyz_s).sum(dim=0)
            # p_s = torch.tensor([100,100,1]).cuda().float().view(3, 1)
            # P_s = K_s.inverse() @ (p_s * 50)
            # P = R_s.inverse() @ (P_s - T_s)
            # P_t = R_t @ P + T_t
            # p_t = K_t @ P_t
            # p_t = p_t / p_t[2]
            # test3 = P_t.transpose(0, 1) @ E @ P_s
            # test4 = p_t.transpose(0, 1) @ F @ p_s

            CD = R.inverse() @ K_t.inverse() @ xyz_t_near
            CA = - T.view(3, 1).repeat(1, CD.shape[1])
            AB = CE = K_s.inverse() @ xyz_s

            CB = CA + AB
            AD = CD - CA

            ACD = torch.cross(CD, CA, dim=0)
            CDE = torch.cross(CE, CD, dim=0)
            
            SACD = torch.sqrt((ACD ** 2).sum(dim=0))
            SCDE = torch.sqrt((CDE ** 2).sum(dim=0))

            depth = SACD / SCDE
            #depth = depth.view(H, W, 1)

            CD_hat = R.inverse() @ K_t.inverse() @ torch.cat([u, v, torch.ones_like(u).cuda()], dim=0)
            DD_hat = CD_hat - CD
            AD_hat = CD_hat - CA

            x = torch.round(x_t_near).long()
            y = torch.round(y_t_near).long()
            valid = torch.logical_and(x >= 0, x <= W-1)
            valid = torch.logical_and(valid, torch.logical_and(y >=0, y <= H-1))
            color_dis = (image_s.view(3, -1) - image_t.view(3, -1)) ** 2
            color_dis = torch.sqrt(color_dis.sum(0))

            cam_dis = ((R_s.inverse() @ T_s - R_t.inverse() @ T_t) ** 2).sum(0, keepdim=True)
            cam_dis = torch.sqrt(cam_dis).repeat(1, H * W)
            
            # g = 10
            # confidence = torch.zeros_like(dis).cuda()
            # confidence[valid] = dis[valid] + cam_dis[valid] + 1
            # confidence[~valid] = (dis[~valid] + cam_dis[~valid] + 1) * g
            # confidence = cam_dis

            len_CA = torch.sqrt((CA ** 2).sum(0, keepdim=True))
            len_AB = torch.sqrt((AB ** 2).sum(0, keepdim=True))
            len_CD = torch.sqrt((CD ** 2).sum(0, keepdim=True))
            len_CB = torch.sqrt((CB ** 2).sum(0, keepdim=True))
            len_AD = torch.sqrt((AD ** 2).sum(0, keepdim=True))
            len_CD_hat = torch.sqrt((CD_hat ** 2).sum(0, keepdim=True))
            len_DD_hat = torch.sqrt((DD_hat ** 2).sum(0, keepdim=True))
            len_AD_hat = torch.sqrt((AD_hat ** 2).sum(0, keepdim=True))
            cos_alpha = (len_CD**2 + len_CA**2 - len_AD**2) / (2 * len_CA * len_CD)
            cos_beta = (len_AB**2 + len_CA**2 - len_CB**2) / (2 * len_CA * len_AB)
            sin_alpha = torch.sqrt(1 - cos_alpha**2)
            sin_beta = torch.sqrt(1 - cos_beta**2)
            confidence = len_CA * sin_beta / ((sin_beta * cos_alpha + sin_alpha * cos_beta) ** 2)

            Ost = - R_t @ R_s.inverse() @ T_s + T_t
            Ost = K_t @ Ost
            AF = R.inverse() @ K_t.inverse() @ (Ost / Ost[2:3, :])
            CF = CA + AF
            len_CF = torch.sqrt((CF ** 2).sum(0, keepdim=True))
            DF = AF - AD
            len_DF = torch.sqrt((DF ** 2).sum(0, keepdim=True))
            cos_theta = (len_CF**2 + len_DF**2 - len_CD**2) / (2 * len_CF * len_DF)
            sin_theta = torch.sqrt(1 - cos_theta**2)
            confidence *= ((sin_alpha * cos_theta + cos_alpha * sin_theta) ** 2) / (len_CF * sin_theta)

            if flow_type == 'dis':
                confidence = torch.abs(confidence)
            elif flow_type == 'grad':
                confidence = torch.abs(confidence)
            elif flow_type == 'disxgrad':
                confidence = torch.abs(confidence) * dis
            # else:
            #     assert False, "Could not recognize flow type!"
            # confidence = torch.abs(confidence) * dis
            # confidence = torch.abs(confidence)
            # confidence = dis

            l =  len_CA * sin_alpha / (sin_beta * cos_alpha + sin_alpha * cos_beta)
            #ll = len_CA / (sin_beta * (cos_alpha/sin_alpha + cos_beta / sin_beta))
            #confidence = dis
            cos_alpha_hat = (len_CD_hat**2 + len_CA**2 - len_AD_hat**2) / (2 * len_CD_hat * len_CA)
            sin_alpha_hat = torch.sqrt(1 - cos_alpha_hat**2)
            #confidence = confidence * torch.acos(cos_alpha_hat)
            #confidence = torch.abs(len_CA * sin_beta / (sin_beta * cos_alpha_hat + sin_alpha_hat * cos_beta))

            d = torch.abs(l / len_AB)

            if flow_type == 'avg':
                if final_confidence is None:
                    final_confidence = confidence
                    final_depth = depth
                    final_mask = mask
                else:
                    final_depth = final_depth + depth
                    final_mask = torch.logical_or(final_mask, mask)
            elif flow_type == 'avg_mask':
                if final_confidence is None:
                    final_confidence = confidence
                    final_depth = torch.where(mask, depth, 0.)
                    final_mask = mask
                    count = torch.where(mask, 1., 0.)
                else:
                    final_depth = torch.where(mask, final_depth + depth, depth)
                    final_mask = torch.logical_or(final_mask, mask)
                    count = torch.where(mask, count + 1, count)
            elif flow_type == 'weight':
                depths.append(depth.unsqueeze(0))
                weights.append(1 / dis)
                if final_confidence is None:
                    final_confidence = confidence
                    final_depth = torch.where(mask, depth, 0.)
                    final_mask = mask
                    count = torch.where(mask, 1., 0.)
                else:
                    final_depth = torch.where(mask, final_depth, depth)
                    final_mask = torch.logical_or(final_mask, mask)
                    count = torch.where(mask, count + 1, count)
            else :
                if final_confidence is None:
                    final_confidence = confidence
                    final_depth = depth
                    final_mask = mask
                else:
                    final_confidence = torch.where(torch.logical_and(final_mask == False, mask == True), confidence, final_confidence)
                    final_depth = torch.where(torch.logical_and(final_mask == False, mask == True), depth, final_depth)
                    final_mask = torch.where(torch.logical_and(final_mask == False, mask == True), mask, final_mask)

                    final_confidence = torch.where(confidence < final_confidence, confidence, final_confidence)
                    final_depth = torch.where(confidence < final_confidence, depth, final_depth)
                    final_mask = torch.where(confidence < final_confidence, mask, final_mask)

                    # final_confidence = torch.where(torch.logical_and(confidence < final_confidence, mask), confidence, final_confidence)
                    # final_depth = torch.where(torch.logical_and(confidence < final_confidence, mask), depth, final_depth)
                    # final_mask = torch.where(torch.logical_and(confidence < final_confidence, mask), mask, final_mask)

                    # final_confidence = torch.where(confidence < final_confidence, confidence, final_confidence)
                    # final_depth = torch.where(confidence < final_confidence, depth, final_depth)
                    # final_mask = torch.where(confidence < final_confidence, mask, final_mask)

        #final_mask = torch.ones((H, W, 1), dtype=torch.bool).cuda()
        #final_mask = final_confidence < valid_dis_threshold
        # image_s = image_s.view(3, -1)
        # mask = torch.logical_or(image_s[0:1, :] < 245./255., image_s[1:2, :] < 245./255.)
        # mask = torch.logical_or(mask, image_s[2:3, :] < 245./255.)
        # #final_depth = torch.where(mask, 200, final_depth)
        # final_mask = torch.logical_and(final_mask, mask)

        # mask = torch.logical_or(image_s[0:1, :] > 10./255., image_s[1:2, :] > 10./255.)
        # mask = torch.logical_or(mask, image_s[2:3, :] > 10./255.)
        # final_mask = torch.logical_and(final_mask, mask)
        #final_mask = confidence < valid_dis_threshold
        if flow_type == 'avg':
            final_depth = final_depth / min(n, len(cams_near))
        if flow_type == 'avg_mask':
            final_depth = final_depth / count
        elif flow_type == 'weight':
            weights = torch.cat(weights, dim=0)
            depths = torch.cat(depths, dim=0)
            weights = torch.softmax(weights, dim=0)
            final_depth = (weights * depths).sum(0).squeeze()
        cams[i].set_flow_depth(final_depth.view(H, W, 1), final_mask.view(H, W, 1))
        

def compute_depth_by_flow1(cams, valid_dis_threshold):
    for i, cam_s in enumerate(cams):
        H, W = cam_s.image_height, cam_s.image_width
        R_s = torch.tensor(cam_s.R.transpose()).cuda().float()
        T_s = torch.tensor(cam_s.T.transpose()).cuda().float().view(3, 1)
        K_s = torch.tensor(cam_s.K.transpose()).cuda().float()
        image_s = cam_s.original_image.cuda()


        n = 3
        cams_near = []
        for j, flow in cam_s.flow:
            cam_t = cams[j]
            R_t = torch.tensor(cam_t.R.transpose()).cuda().float()
            T_t = torch.tensor(cam_t.T.transpose()).cuda().float().view(3, 1)
            cam_dis = ((R_s.inverse() @ T_s - R_t.inverse() @ T_t) ** 2).sum(0)
            cam_dis = torch.sqrt(cam_dis)
            cams_near.append((j, float(cam_dis), flow))
        cams_near = sorted(cams_near, key=lambda x:x[1])



        id_dis = 100000
        final_confidence = None
        final_depth = None
        final_mask = None
        for j, _, flow in cams_near[:min(n, len(cams_near))]:
            # if abs(i - j) > 1:
            #     continue
            flow = torch.tensor(flow).cuda()
            cam_t = cams[j]
            image_t = cam_t.original_image.cuda()

            R_t = torch.tensor(cam_t.R.transpose()).cuda().float()
            T_t = torch.tensor(cam_t.T.transpose()).cuda().float().view(3, 1)
            K_t = torch.tensor(cam_t.K.transpose()).cuda().float()

            coords_y = torch.arange(0, H).cuda()
            coords_x = torch.arange(0, W).cuda()
            y, x = torch.meshgrid(coords_y, coords_x)
            y, x = y.unsqueeze(2), x.unsqueeze(2)
            xyz_s = torch.cat([x, y, torch.ones_like(x).cuda().float()], dim=0).view(3, -1)
            

            R = R_t @ R_s.inverse()
            T = R_s @ R_t.inverse() @ T_t - T_s
            T = T.squeeze()
            S = torch.zeros((3, 3)).cuda()
            S[0, 1] = - T[2]
            S[1, 0] = T[2]
            S[0, 2] = T[1]
            S[2, 0] = - T[1]
            S[1, 2] = - T[0]
            S[2, 1] = T[0]
            E = R @ S
            F = K_t.inverse().transpose(0, 1) @ E @ K_s.inverse()

            epipolar_line = F @ xyz_s
            a = epipolar_line[0:1]
            b = epipolar_line[1:2]
            c = epipolar_line[2:3]

            u = (x.contiguous().view(H, W, 1) + flow[:, :, :1]).contiguous().view(1, -1)
            v = (y.contiguous().view(H, W, 1) + flow[:, :, 1:2]).contiguous().view(1, -1)

            x_t_near = (b * b * u - a * b * v - a * c) / (a ** 2 + b ** 2)
            y_t_near = (a * a * v - a * b * u - b * c) / (a ** 2 + b ** 2)
            xyz_t_near = torch.cat([x_t_near, y_t_near, torch.ones_like(x_t_near).cuda()], dim=0)

            dis = torch.abs((a * u + b * v + c) / torch.sqrt(a ** 2 + b ** 2))
            dis_max = dis.max()
            mask = (dis < valid_dis_threshold)
            #mask = torch.ones_like(mask, dtype=torch.bool).cuda()
            # xyz_t = torch.cat([u, v, torch.ones_like(u).cuda()], dim=0)
            # mask_num = mask.sum()
            # test1 = ((F.transpose(0, 1) @ xyz_t_near) * xyz_s).sum(dim=0)
            # test2 = ((F.transpose(0, 1) @ xyz_t) * xyz_s).sum(dim=0)
            # p_s = torch.tensor([100,100,1]).cuda().float().view(3, 1)
            # P_s = K_s.inverse() @ (p_s * 50)
            # P = R_s.inverse() @ (P_s - T_s)
            # P_t = R_t @ P + T_t
            # p_t = K_t @ P_t
            # p_t = p_t / p_t[2]
            # test3 = P_t.transpose(0, 1) @ E @ P_s
            # test4 = p_t.transpose(0, 1) @ F @ p_s

            CD = R.inverse() @ K_t.inverse() @ xyz_t_near
            CA = - T.view(3, 1).repeat(1, CD.shape[1])
            AB = CE = K_s.inverse() @ xyz_s

            CB = CA + AB
            AD = CD - CA

            ACD = torch.cross(CD, CA, dim=0)
            CDE = torch.cross(CE, CD, dim=0)
            
            SACD = torch.sqrt((ACD ** 2).sum(dim=0))
            SCDE = torch.sqrt((CDE ** 2).sum(dim=0))

            depth = SACD / SCDE
            #depth = depth.view(H, W, 1)

            CD_hat = R.inverse() @ K_t.inverse() @ torch.cat([u, v, torch.ones_like(u).cuda()], dim=0)
            DD_hat = CD_hat - CD
            AD_hat = CD_hat - CA

            x = torch.round(x_t_near).long()
            y = torch.round(y_t_near).long()
            valid = torch.logical_and(x >= 0, x <= W-1)
            valid = torch.logical_and(valid, torch.logical_and(y >=0, y <= H-1))
            color_dis = (image_s.view(3, -1) - image_t.view(3, -1)) ** 2
            color_dis = torch.sqrt(color_dis.sum(0))

            cam_dis = ((R_s.inverse() @ T_s - R_t.inverse() @ T_t) ** 2).sum(0, keepdim=True)
            cam_dis = torch.sqrt(cam_dis).repeat(1, H * W)
            
            # g = 10
            # confidence = torch.zeros_like(dis).cuda()
            # confidence[valid] = dis[valid] + cam_dis[valid] + 1
            # confidence[~valid] = (dis[~valid] + cam_dis[~valid] + 1) * g
            # confidence = cam_dis

            len_CA = torch.sqrt((CA ** 2).sum(0, keepdim=True))
            len_AB = torch.sqrt((AB ** 2).sum(0, keepdim=True))
            len_CD = torch.sqrt((CD ** 2).sum(0, keepdim=True))
            len_CB = torch.sqrt((CB ** 2).sum(0, keepdim=True))
            len_AD = torch.sqrt((AD ** 2).sum(0, keepdim=True))
            len_CD_hat = torch.sqrt((CD_hat ** 2).sum(0, keepdim=True))
            len_DD_hat = torch.sqrt((DD_hat ** 2).sum(0, keepdim=True))
            len_AD_hat = torch.sqrt((AD_hat ** 2).sum(0, keepdim=True))
            cos_alpha = (len_CD**2 + len_CA**2 - len_AD**2) / (2 * len_CA * len_CD)
            cos_beta = (len_AB**2 + len_CA**2 - len_CB**2) / (2 * len_CA * len_AB)
            sin_alpha = torch.sqrt(1 - cos_alpha**2)
            sin_beta = torch.sqrt(1 - cos_beta**2)
            confidence = torch.abs(len_CA * sin_beta / (sin_beta * cos_alpha + sin_alpha * cos_beta))
            l =  len_CA * sin_alpha / (sin_beta * cos_alpha + sin_alpha * cos_beta)
            #ll = len_CA / (sin_beta * (cos_alpha/sin_alpha + cos_beta / sin_beta))
            #confidence = dis
            cos_alpha_hat = (len_CD_hat**2 + len_CA**2 - len_AD_hat**2) / (2 * len_CD_hat * len_CA)
            sin_alpha_hat = torch.sqrt(1 - cos_alpha_hat**2)
            #confidence = confidence * torch.acos(cos_alpha_hat)
            #confidence = torch.abs(len_CA * sin_beta / (sin_beta * cos_alpha_hat + sin_alpha_hat * cos_beta))

            d = torch.abs(l / len_AB)

            if final_confidence is None:
                final_confidence = confidence
                final_depth = depth
                final_mask = mask
            else:
                final_confidence = torch.where(torch.logical_and(confidence < final_confidence, mask), confidence, final_confidence)
                final_depth = torch.where(torch.logical_and(confidence < final_confidence, mask), depth, final_depth)
                final_mask = torch.where(torch.logical_and(confidence < final_confidence, mask), mask, final_mask)

        #final_mask = torch.ones((H, W, 1), dtype=torch.bool).cuda()
        cams[i].set_flow_depth(final_depth.view(H, W, 1), final_mask.view(H, W, 1))


def compute_depth_by_flow2(cams):
    for i, cam_s in enumerate(cams):
        id_dis = 100000
        for j, flow in cam_s.flow:
            flow = torch.tensor(flow).cuda()
            cam_t = cams[j]

            R_s = torch.tensor(cam_s.R.transpose()).cuda().float()
            T_s = torch.tensor(cam_s.T.transpose()).cuda().float().view(3, 1)
            K_s = torch.tensor(cam_s.K.transpose()).cuda().float()
            R_t = torch.tensor(cam_t.R.transpose()).cuda().float()
            T_t = torch.tensor(cam_t.T.transpose()).cuda().float().view(3, 1)
            K_t = torch.tensor(cam_t.K.transpose()).cuda().float()
            H, W = cam_s.image_height, cam_s.image_width
            image = cam_s.original_image.cuda()

            coords_y = torch.arange(0, H).cuda()
            coords_x = torch.arange(0, W).cuda()
            y, x = torch.meshgrid(coords_y, coords_x)
            y, x = y.unsqueeze(2), x.unsqueeze(2)
            xyz_s = torch.cat([x, y, torch.ones_like(x).cuda().float()], dim=0).view(3, -1)
            

            R = R_t @ R_s.inverse()
            T = R_s @ R_t.inverse() @ T_t - T_s
            T = T.squeeze()
            S = torch.zeros((3, 3)).cuda()
            S[0, 1] = - T[2]
            S[1, 0] = T[2]
            S[0, 2] = T[1]
            S[2, 0] = - T[1]
            S[1, 2] = - T[0]
            S[2, 1] = T[0]
            E = R @ S
            F = K_t.inverse().transpose(0, 1) @ E @ K_s.inverse()

            epipolar_line = F @ xyz_s
            a = epipolar_line[0:1]
            b = epipolar_line[1:2]
            c = epipolar_line[2:3]

            u = (x.contiguous().view(H, W, 1) + flow[:, :, :1]).contiguous().view(1, -1)
            v = (y.contiguous().view(H, W, 1) + flow[:, :, 1:2]).contiguous().view(1, -1)

            x_t_near = (b * b * u - a * b * v - a * c) / (a ** 2 + b ** 2)
            y_t_near = (a * a * v - a * b * u - b * c) / (a ** 2 + b ** 2)
            xyz_t_near = torch.cat([x_t_near, y_t_near, torch.ones_like(x_t_near).cuda()], dim=0)
            xyz_t = torch.cat([u, v, torch.ones_like(u).cuda()], dim=0)

            dis = torch.abs((a * u + b * v + c) / torch.sqrt(a ** 2 + b ** 2))
            dis_max = dis.max()
            mask = (dis < 1.)
            mask_num = mask.sum()
            test1 = ((F.transpose(0, 1) @ xyz_t_near) * xyz_s).sum(dim=0)
            test2 = ((F.transpose(0, 1) @ xyz_t) * xyz_s).sum(dim=0)
            p_s = torch.tensor([100,100,1]).cuda().float().view(3, 1)
            P_s = K_s.inverse() @ (p_s * 50)
            P = R_s.inverse() @ (P_s - T_s)
            P_t = R_t @ P + T_t
            p_t = K_t @ P_t
            p_t = p_t / p_t[2]
            test3 = P_t.transpose(0, 1) @ E @ P_s
            test4 = p_t.transpose(0, 1) @ F @ p_s


            CD = R.inverse() @ K_t.inverse() @ xyz_t_near
            CA = - T.view(3, 1).repeat(1, CD.shape[1])
            CE = K_s.inverse() @ xyz_s

            ACD = torch.cross(CD, CA, dim=0)
            CDE = torch.cross(CE, CD, dim=0)
            
            SACD = torch.sqrt((ACD ** 2).sum(dim=0))
            SCDE = torch.sqrt((CDE ** 2).sum(dim=0))

            depth = SACD / SCDE
            depth = depth.view(H, W, 1)
            
            if abs(i - j) < id_dis:
                id_dis = abs(i - j)
                cams[i].set_flow_depth(depth, mask.view(H, W, 1))