# -- coding:utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool
class Resnet50RoiHead(nn.Module):
    def __init__(self,n_class,roi_size,spatial_scale,classifier):
        super(Resnet50RoiHead,self).__init__()
        self.classifier = classifier
        #回归预坐标测
        self.cls_loc = nn.Linear(2048,n_class*4)
        # 分类预测
        self.score = nn.Linear(2048,n_class)
        self.roi = RoIPool((roi_size,roi_size),spatial_scale)

    def forward(self,x,rois,roi_indices,img_size):
        n,_,_,_ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:,[0,2]]=rois[:,[0,2]]/img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        fc7 = self.classifier(pool)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

