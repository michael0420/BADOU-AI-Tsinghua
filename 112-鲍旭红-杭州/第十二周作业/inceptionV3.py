#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 20:45
# @Author  : bxh
# @Email   : 573878341@qq.com
# @File    : inceptionV3.py



import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,BatchNormalization,Input,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPool2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

def conv2d_bn(x,filters,num_row,num_col,strides=(1,1),padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters,(num_row,num_col),strides=strides,padding=padding,use_bias=False,name=conv_name)(x)
    x = BatchNormalization(scale=False,name=bn_name)(x)
    x = Activation('relu',name=name)(x)
    return x



def InceptionV3(input_shape = [224,224,3],classes = 2):
    img_input = Input(shape=input_shape)
    x = conv2d_bn(img_input,32,3,3,strides=(2,2),padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3,3),strides=(2,2))(x)

#     -------------------
# Block1 35*35
# ------------------------
#Block1 part1
#35*35*192 -> 35*35*256
    branch1x1 = conv2d_bn(x,64,1,1)
    branch5x5 = conv2d_bn(x,48,1,1)
    branch5x5 = conv2d_bn(branch5x5,64,5,5)

    branch3x3db1 = conv2d_bn(x,64,1,1)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,32,1,1)

    x = layers.concatenate([branch1x1,branch5x5,branch3x3db1,branch_pool],axis=3,name='mixed0')


#     -------------------
# Block1 35*35
# ------------------------
#Block1 part2
#35*35*256 -> 35*35*288
    branch1x1 = conv2d_bn(x,64,1,1)

    branch5x5 = conv2d_bn(x,48,1,1)
    branch5x5 = conv2d_bn(branch5x5,64,5,5)

    branch3x3db1 = conv2d_bn(x,64,1,1)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,64,1,1)

    #64+64+96+64 = 288
    x = layers.concatenate([branch1x1,branch5x5,branch3x3db1,branch_pool],axis=3,name='mixed1')

    #     -------------------
    # Block1 35*35
    # ------------------------
    # Block1 part3
    # 35*35*288 -> 35*35*288
    branch1x1 = conv2d_bn(x,64,1,1)

    branch5x5 = conv2d_bn(x,48,1,1)
    branch5x5 = conv2d_bn(branch5x5,64,5,5)

    branch3x3db1 = conv2d_bn(x,64,1,1)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,64,1,1)

    x = layers.concatenate([branch1x1,branch5x5,branch3x3db1,branch_pool],axis=3,name='mixed2')

    #     -------------------
    # Block2 17*17
    # ------------------------
    # Block2 part1
    # 35*35*288 -> 17*17*768
    branch3x3 = conv2d_bn(x,384,3,3,strides=(2,2),padding='valid')

    branch3x3db1 = conv2d_bn(x,64,1,1)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3,strides=(2,2),padding='valid')

    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)

    x = layers.concatenate([branch3x3,branch3x3db1,branch_pool],axis=3,name='mixed3')

    #     -------------------
    # Block2 17*17
    # ------------------------
    # Block2 part2
    # 17*17*768 -> 17*17*768
    branch1x1 = conv2d_bn(x,192,1,1)

    branch7x7 = conv2d_bn(x,128,1,1)
    branch7x7 = conv2d_bn(branch7x7,128,1,7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7db1 = conv2d_bn(x,128,1,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,128,7,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,128,1,7)
    branch7x7db1 = conv2d_bn(branch7x7db1,128,7,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,192,1,7)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,192,1,1)
    x = layers.concatenate([branch1x1,branch7x7,branch7x7db1,branch_pool],axis=3,name='mixed4')

    # Block2 part3 and part4
    # 17*17*768 -> 17*17*768->17*17*768
    for i in range(2):
        branch1x1 = conv2d_bn(x,192,1,1)
        
        branch7x7 = conv2d_bn(x,160,1,1)
        branch7x7 = conv2d_bn(branch7x7,160,1,7)
        branch7x7 = conv2d_bn(branch7x7,192,7,1)
        
        branch7x7db1 = conv2d_bn(x,160,1,1)
        branch7x7db1 = conv2d_bn(branch7x7db1,160,7,1)
        branch7x7db1 = conv2d_bn(branch7x7db1,160,1,7)
        branch7x7db1 = conv2d_bn(branch7x7db1,160,7,1)
        branch7x7db1 = conv2d_bn(branch7x7db1,192,1,7)
        
        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2d_bn(branch_pool,192,1,1)
        
        x = layers.concatenate([branch1x1,branch7x7,branch7x7db1,branch_pool],axis=3,name='mixed'+str(5+i))

    # Block2 part5
    # 17*17*768 -> 17*17*768->17*17*768
    branch1x1 = conv2d_bn(x,192,1,1)
    
    branch7x7 = conv2d_bn(x,192,1,1)
    branch7x7 = conv2d_bn(branch7x7,192,1,7)
    branch7x7 = conv2d_bn(branch7x7,192,7,1)
    
    branch7x7db1 = conv2d_bn(x,192,1,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,192,7,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,192,1,7)
    branch7x7db1 = conv2d_bn(branch7x7db1,192,7,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,192,1,7)
    
    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,192,1,1)
    
    x = layers.concatenate([branch1x1,branch7x7,branch7x7db1,branch_pool],axis=3,name='mixed7')

    # Block3 part1 8*8
    # 17*17*768 -> 17*17*768->17*17*1280
    branch3x3 = conv2d_bn(x,192,1,1)
    branch3x3 = conv2d_bn(branch3x3,320,3,3,strides=(2,2),padding='valid')
    
    branch7x7x3 = conv2d_bn(x,192,1,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,1,7)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,7,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,3,3,strides=(2,2),padding='valid')

    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)
    x = layers.concatenate([branch3x3,branch7x7x3,branch_pool],axis=3,name='mixed8')

    # Block3 part1 8*8
    # 17*17*1280 -> 17*17*768->17*17*2048
    for i in range(2):
        branch1x1 = conv2d_bn(x,320,1,1)

        branch3x3 = conv2d_bn(x,384,1,1)
        branch3x3_1 = conv2d_bn(branch3x3,384,1,3)
        branch3x3_2 = conv2d_bn(branch3x3,384,3,1)
        branch3x3 = layers.concatenate([branch3x3_1,branch3x3_2],axis=3,name='mixed9_'+str(i))

        branch3x3db1 = conv2d_bn(x,448,1,1)
        branch3x3db1 = conv2d_bn(branch3x3db1,384,3,3)
        branch3x3db1_1 = conv2d_bn(branch3x3db1,384,1,3)
        branch3x3db1_2 = conv2d_bn(branch3x3db1,384,3,1)
        branch3x3db1 = layers.concatenate([branch3x3db1_1,branch3x3db1_2],axis=3,name='mixed'+str(9+i))

    #平均池化后全连接
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes,activation='softmax',name='predictions')(x)

    inputs = img_input
    model = Model(inputs,x,name='inception_v3')
    return model






