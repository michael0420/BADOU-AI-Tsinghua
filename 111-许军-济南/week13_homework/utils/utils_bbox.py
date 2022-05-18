# -- coding:utf-8 --
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.ops import nms

def loc2bbox(src_bbox,loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0,4),dtype=loc.dtype)

    src_width=torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height=torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x=torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y=torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height


    dx = loc[:,0::4]
    dy = loc[:,1::4]
    dw = loc[:,2::4]
    dh = loc[:,3::4]

    ctr_x=dx * src_width + src_ctr_x
    ctr_y=dy * src_height + src_ctr_y
    w=torch.exp(dw) * src_width
    h=torch.exp(dh) * src_height


    # 对调整后的进行转变调整成左上角右下角的格式
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:,0::4] = ctr_x - 0.5 * w
    dst_bbox[:,1::4] = ctr_y - 0.5 * h
    dst_bbox[:,2::4] = ctr_x + 0.5 * w
    dst_bbox[:,3::4] = ctr_y + 0.5 * h
    return dst_bbox