import torch.nn as nn

from week13_homework.nets.classifier import Resnet50RoiHead
from week13_homework.nets.resnet50 import resnet50
from week13_homework.nets.rpn import RegionProposalNetwork



class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,  
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        #---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        #---------------------------------#

        if backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet50RoiHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )
            
    def forward(self, x, scale=1.):
        #---------------------------------#
        #   计算输入图片的大小
        #---------------------------------#
        img_size        = x.shape[2:]
        #---------------------------------#
        #   利用主干网络提取特征
        #---------------------------------#
        base_feature    = self.extractor.forward(x)

        #---------------------------------#
        #   获得建议框
        #---------------------------------#
        _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
        #---------------------------------------#
        #   获得classifier的分类结果和回归结果
        #---------------------------------------#
        roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
