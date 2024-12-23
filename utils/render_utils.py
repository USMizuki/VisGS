import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np


def apply_depth_colormap(gray, minmax=[0.1, 100.0], cmap=cv2.COLORMAP_JET):
    """
    Input:
        gray: gray image, tensor/numpy, (H, W)
    Output:
        depth: (3, H, W), tensor
    """
    if type(gray) is not np.ndarray:
        gray = gray.detach().cpu().numpy().astype(np.float32)
    gray = gray.squeeze()
    assert len(gray.shape) == 2
    x = np.nan_to_num(gray)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive value
        ma = np.max(x)
    else:
        mi, ma = minmax
        x = np.clip(x, mi, ma)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    # TODO
    x = 1 - x  # reverse the colormap
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(cv2.applyColorMap(x, cmap))
    x = T.ToTensor()(x)  # (3, H, W)
    return x