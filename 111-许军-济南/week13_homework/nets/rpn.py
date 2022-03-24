# -- coding:utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from week13_homework.utils import anchors,utils_bbox
from torchvision.ops import nms
import numpy as np

class ProposalCreater():
    def __init__(self,mode,nms_iou = 0.7,n_train_pre_nms = 12000,n_train_post_nms = 600,
                 n_test_pre_nms = 3000,n_test_post_nms = 300,min_size = 16):

        self.mode=mode
        self.nms_iou=nms_iou
        self.n_train_pre_nms=n_train_pre_nms
        self.n_train_post_nms=n_train_post_nms
        self.n_test_pre_nms=n_test_pre_nms
        self.n_test_post_nms=n_test_post_nms
        self.min_size=min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.) :
        if self.mode == "training" :
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms=self.n_test_pre_nms
            n_post_nms=self.n_test_post_nms
        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()

        # 将rpn网络结果转换成建议框
        roi=utils_bbox.loc2bbox(anchor, loc)
        # 防止建议框超出图像边缘
        roi[:, [0, 2]]=torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])  # 限制到某个范围内
        roi[:, [1, 3]]=torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])
        # 宽高的值不可以小于16
        min_size=self.min_size * scale
        # 筛选合条件的框
        keep=torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) > min_size))[0]
        roi=roi[keep, :]
        score=score[keep]
        # 根据得分取出建议框
        order=torch.argsort(score, descending=True)  # 按从大到小排序
        if n_pre_nms > 0 :
            order = order[:n_pre_nms]
        roi = roi[order,:]
        score = score[order]

        keep = nms(roi,score,self.nms_iou)
        keep = keep[:n_post_nms]
        roi = roi[keep]

        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(self,in_channels = 512,out_channels = 512,ratios=[0.5,1,2],anchor_scales=[8,16,32],feat_stride = 16,mode="training"):
        super(RegionProposalNetwork,self).__init__()
        self.feat_stride = feat_stride
        #生成基础先验框
        self.anchor_base = anchors.generate_anchor_base(anchor_scales=anchor_scales,ratios=ratios)
        n_anchor = self.anchor_base.shape[0]

        # 先经过3*3的卷积
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        # 2*9的分支，预测框内是否有物体
        self.score = nn.Conv2d(in_channels=out_channels,out_channels=2*n_anchor,kernel_size=1,stride=1,padding=0)
        # 回归预测
        self.loc = nn.Conv2d(in_channels=out_channels,out_channels=n_anchor*4,kernel_size=1,stride=1,padding=0)
        self.proposal_layer = ProposalCreater(mode)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)




    def forward(self,x,img_size,scale=1.):
        n,c,h,w = x.shape


        x = F.relu(self.conv1(x))
        # 分类的支线
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0,2,3,1).contiguous().view(n,-1,2)
        rpn_softmax_scores = F.softmax(rpn_scores,dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:,:,1].contiguous()#包含物体的概率
        rpn_fg_scores = rpn_fg_scores.view(n,-1)

        # 回归的支线
        rpn_log = self.loc(x)
        rpn_logs=rpn_log.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # 生成先验框 一个网格点对应9个框
        anchor = anchors._enumerate_shifted_anchor(np.array(self.anchor_base),self.feat_stride,h,w)
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_logs[i],rpn_fg_scores[i],anchor,img_size,scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = torch.cat(rois,dim=0)
        roi_indices = torch.cat(roi_indices,dim=0)
        return rpn_logs,rpn_scores,rois,roi_indices,anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()



