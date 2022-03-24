# -- coding:utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecodeBox(nn.Module):
    def __init__(self,anchors,num_classes,img_size):
        super(DecodeBox,self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self,input):
        # 一共多少张图片
        batch_size = input.size(0)
        # 输入的宽和高
        input_height = input.size(2)
        input_weight = input.size(3)

        # 一个特征点对应原图多少个像素点
        stride_h = self.img_size[1]/input_height
        stride_w = self.img_size[0]/input_weight

        # 把先验框的尺寸调整featuremap大小
        scale_anchors = [(anchor_width/stride_w,anchor_height/stride_h) for anchor_width,anchor_height in self.anchors]
        #调整预测结果的维度
        prediction = input.view(batch_size,self.num_classes,self.bbox_attrs,input_height,input_weight).permute(0,1,3,4,2).contiguous()

        # 先验框中心位置的调整参数
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        w = prediction[...,2]
        h = prediction[...,3]
        # 获得置信度
        conf = torch.sigmoid(prediction[...,4])
        # 获得种类置信度
        pred_cls = torch.sigmoid(prediction[...,5:])
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0,input_weight-1,input_weight).repeat(input_weight,1).repeat(batch_size*self.num_anchors,
                                1,1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).repeat(batch_size * self.num_anchors,
            1, 1).view(x.shape).type(FloatTensor)

        #生成先验框的宽高
        anchor_w = FloatTensor(scale_anchors).index_select(1,LongTensor([0]))
        anchor_h = FloatTensor(scale_anchors).index_select(1,LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size,1).repeat(1,1,input_height*input_weight).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size,1).repeat(1,1,input_weight*input_height).view(h.shape)

        # 计算调整后的先验框的宽高
        pred_boxes = FloatTensor(prediction[...,:4].shape)
        pred_boxes[...,0] = x.data + grid_x
        pred_boxes[...,1]= y.data + grid_y
        pred_boxes[...,2]= torch.exp(w.data)*anchor_w
        pred_boxes[...,3]= torch.exp(h.data)*anchor_h

        #用于输出调整为416*416的大小
        _scale = torch.Tensor([stride_w,stride_h]*2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size,-1,4)*_scale,conf.view(batch_size,-1,1),pred_cls.view(batch_size,-1,self.num_classes)),-1)
        return output.data
