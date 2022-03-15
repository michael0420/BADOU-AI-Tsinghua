#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 20:56
# @Author  : bxh
# @Email   : 573878341@qq.com
# @File    : mobilnet.py

import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,AveragePooling2D,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling1D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def relu6(x):
    return K.relu(x,max_value=6)


def _conv_block(inputs,filters,kernel=(3,3),strides=(1,1)):
    x = Conv2D(filters,kernel,padding='same',use_bias=False,strides=strides,name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6,name='conv1_relu')(x)



def _depthwise_conv_block(inputs,pointwise_conv_filters,
                          depth_multipliter=1,strides=(1,1),block_id = 1):
    x = DepthwiseConv2D((3,3),padding='same',depth_multiplier=depth_multipliter,strides=strides,use_bias=False,name='conv_dw_%d'%block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn'%block_id)(x)
    x = Activation(relu6,name='conv_dw_%d_relu'%block_id)(x)

    x = Conv2D(pointwise_conv_filters,(1,1),padding='same',use_bias=False,strides=(1,1),name='conv_pw_%d'%block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn'%block_id)(x)
    return Activation(relu6,name='conv_pw_%d_relu'%block_id)(x)


def MobileNet(input_shape=[224,224,3],
              depth_multiplieter=1,
              dropout=0.8,
              classes=2):
    img_input = Input(shape=input_shape)
    #224,224,3->112,112,32
    x = _conv_block(img_input,32,strides=(2,2))

    #112,112,32->112,112,64
    x = _depthwise_conv_block(x,64,depth_multiplieter,block_id=1)

    #112,112,64->56,56,128
    x = _depthwise_conv_block(x,128,depth_multiplieter,strides=(2,2),block_id=2)

    #56,56,128->56,56,128
    X = _depthwise_conv_block(x,128,depth_multiplieter,block_id=3)

    #56,56,128->28,28,256
    x = _depthwise_conv_block(x,256,depth_multiplieter,strides=(2,2),block_id=4)

    #28,28,256->28,28,256
    x = _depthwise_conv_block(x,256,depth_multiplieter,block_id=5)

    #28,28,256->14,14,512
    x = _depthwise_conv_block(x,512,depth_multiplieter,strides=(2,2),block_id=6)

    #14,14,512->14,14,512
    x = _depthwise_conv_block(x,512,depth_multiplieter,block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplieter, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplieter, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplieter, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplieter, block_id=11)

    #14,14,512->7,7,1024
    x = _depthwise_conv_block(x,1024,depth_multiplieter,strides=(2,2),block_id=12)
    x = _depthwise_conv_block(x,1024,depth_multiplieter,block_id=13)

    #7,7,1024->1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024),name='reshape_1')(x)
    x = Dropout(dropout,name='dropout')(x)
    x = Conv2D(classes,(1,1),padding='same',name='conv_preds')(x)
    x = Activation('softmax',name='act_softmax')(x)
    x = Reshape((classes,),name='reshape_2')(x)

    inputs =  img_input

    model = Model(inputs,x,name='mobilenet_1_0_224_tf')
    return model




