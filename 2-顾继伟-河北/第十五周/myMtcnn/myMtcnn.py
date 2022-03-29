#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 19:17
# @Author  : ystbr
# @FileName: myMtcnn.py

from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import utils
import cv2

def myPnet(weight_path):
    input = Input(shape=[None,None,3])

    x = Conv2D(10,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='Prelu1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu3')(x)

    # ------------------------------------------------------------------------------------------------------------------
    # 2个分类数据，4个边框回归数据，少了10个人脸关键点数据
    classify = Conv2D(2,(1,1),activation='softmax',name='conv4_1')(x)
    bbox_regression = Conv2D(4,(1,1),name='conv4_2')(x)
    model = Model([input],[classify,bbox_regression])
    model.load_weights(weight_path,by_name=True)
    return model

def myRnet(weight_path):
    input = Input(shape=[24,24,3])

    x = Conv2D(28,(3,3),strides=1,padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='Prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2,padding='same')(x)
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu3')(x)
    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    x = Dense(128,name='conv4')(x)
    x = PReLU(name='Prelu4')(x)

    # ------------------------------------------------------------------------------------------------------------------
    # classify改成了Dense(),其余等于重复Pnet获取2分类和4边框的操作
    classify = Dense(2, activation='softmax', name='conv5_1')(x)
    bbox_regression = Dense(4, name='conv5_2')(x)
    model = Model([input], [classify,bbox_regression])
    model.load_weights(weight_path, by_name=True)
    return model

def myOnet(weight_path):
    input = Input(shape=[48,48,3])

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='Prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu3')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(128, (3, 3), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='Prelu4')(x)
    # 重复Rnet倒转维度的操作,Onet还比Rnet多了一层conv
    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='Prelu5')(x)
    classify = Dense(2, activation='softmax', name='conv6_1')(x)
    bbox_regression = Dense(4, name='conv6_2')(x)
    model = Model([input], [classify,bbox_regression])
    model.load_weights(weight_path, by_name=True)
    return model

class myMtcnn():
    def __init__(self):
        self.Pnet = myPnet('pnet.h5')
        self.Rnet = myPnet('rnet.h5')
        self.Onet = myPnet('onet.h5')

    def detect_face(self,img,threshold):
        copy_img = (img.copy()-127.5)/127.5
        originh,originw,_ = copy_img.shape
        scales = utils.calculateScales(img)
        out = []

        # --------------------------------------------------------------------------------------------------------------
        # 图像金字塔，通过缩放实现,衔接Pnet
        for scale in scales:
            hs = int(originh*scale)
            ws = int(originw*scale)
            scale_img = cv2.resize(copy_img,(hs,ws))
            inputs = scale_img.reshape(1, *scale_img.shape)
            output = self.Pnet.predict(inputs)
            out.append(output)

        img_num = len(scales)
        rectangles = []
        for i in range(img_num):
            class_probability = out[i][0][0][:,:,1]
            roi = out[i][1][0]

            out_h,out_w = class_probability.shape
            out_side = max(out_h,out_w)
            print(class_probability)

            # 不明觉厉
            rectangle = utils.detect_face_12net(class_probability, roi, out_side, 1 / scales[i], originw, originh,
                                                threshold[0])
            rectangles.extend(rectangle)

        rectangles = utils.NMS(rectangles,0.7)
        if len(rectangles) == 0:
            return rectangles

        # --------------------------------------------------------------------------------------------------------------
        # Rnet部分
        predict_24_batch = []
        for rectangle in rectangles:
            # 裁切
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            # 缩放到24x24
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        class_probability = out[0]
        class_probability = np.array(class_probability)
        roi_probability = out[1]
        roi_probability = np.array(roi_probability)
        rectangles = utils.filter_face_24net(class_probability, roi_probability, rectangles, originw, originh, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        # --------------------------------------------------------------------------------------------------------------
        # Onet部分
        # 与Rnet输入部分相同
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        class_probability = output[0]
        roi_probability = output[1]
        pts_prob = output[2] #不明朗
        rectangles = utils.filter_face_48net(class_probability, roi_probability, pts_prob, rectangles, originw, originh, threshold[2])
        return rectangles
















