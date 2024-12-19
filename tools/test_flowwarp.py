import numpy as np
import cv2

from utils.flow_utils import readFlow

img1 = cv2.imread('dataset/nerf_llff_data/fern/3_views/flow_image/things/12_0.png')
# img2 = np.zeros_like(img1)

flow = readFlow('dataset/nerf_llff_data/fern/3_views/flow/things/image001.flo')

h,w = img1.shape[0], img1.shape[1]

y = np.arange(0, h)
x = np.arange(0, w)
x,y = np.meshgrid(x,y)

xy = np.stack([x,y], axis=-1) + flow

new = np.round(xy).astype(np.int64).reshape(-1, 2)

# valid = np.logical_and(new[:, 0] >= 0, new[:, 0] <= w-1)
# valid = np.logical_and(valid, np.logical_and(new[:, 1] >=0, new[:, 1] <= h-1))
# new = new[valid]

w = new[:, 0].max() - new[:, 0].min() + 1
h = new[:, 1].max() - new[:, 1].min() + 1
new[:, 0] = new[:, 0] - new[:, 0].min()
new[:, 1] = new[:, 1] - new[:, 1].min()

img2 = np.zeros((h, w, 3))

img2[new[:, 1], new[:, 0], :] = img1.reshape(-1, 3)

cv2.imwrite('flow_warp.png', img2)