# -- coding:utf-8 --
import numpy as np

def generate_anchor_base(base_size = 16,ratios=[0.5,1,2],anchor_scales = [8,16,32]):
    anchor_base = np.zeros((3*len(anchor_scales),4),dtype=np.float32)
    for i in range(len(anchor_scales)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1./ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index,0] = - h/2
            anchor_base[index,1] = - w/2
            anchor_base[index,2] = h/2
            anchor_base[index,3] = w/2
    return anchor_base

#将anchor映射到featuremap的每个网格点上
def _enumerate_shifted_anchor(anchor_base,feat_stride,height,width):
    shift_x = np.arange(0,width*feat_stride,feat_stride)
    shift_y = np.arange(0,height* feat_stride, feat_stride)
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)
    shift = np.stack((shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel()),axis=1)
    # 每个网格点对顶9个anchor
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = shift.reshape((K,1,4))+anchor_base.reshape((1,A,4))
    anchors = anchor.reshape((K*A,4)).astype(np.float32)
    return anchors

