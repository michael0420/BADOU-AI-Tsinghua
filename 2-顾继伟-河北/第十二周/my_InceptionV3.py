from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

def conv_bn(x,filters,rows,cols,strides=(1,1),padding='same',name=None):   #定义基本块即带批量归一化的卷积层
    """
    :param x: 输入层
    :param filters: 卷积核数量
    :param rows: 卷积核的大小，长
    :param cols: 卷积核的大小，宽
    :param strides: 步长
    :param paddings: 填充方式
    :param name: 图名，在可视化视图中显示网络结构
    :return:
    """
    # 如果该图没有名字就自动加上后缀，否则为空
    if name is not None:
        bn_name = name+'_bn'
        conv_name = name+'_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters,(rows,cols),strides=(1,1),padding='same',use_bias=False,name=conv_name)(x)
    x = BatchNormalization(scale=False,name=bn_name)(x)
    x = Activation(name=name)(x)
    return x

# 定义完整的InceptionV3结构
def my_InceptionV3(input_shape=[299,299,3],classes=1000):
    img_input = Input(input_shape)  #完成输入
    x = conv_bn(img_input,32,3,3,strides=(2,2),padding='valid')
    x = conv_bn(x,32,3,3,padding='valid')
    x = conv_bn(x,64,3,3)
    x = MaxPooling2D((3,3),strides=(2,2))(x)
    x = conv_bn(x,80,3,3,padding='valid')
    x = conv_bn(x,192,3,3,padding='valid')
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    # 第一个block,包括1个1*1,1个5*5,2个3*3,1个3*3池化;;1*1卷积通常在其他卷积之前,在3*3池化之前
    # 第一个block分3个module,第二个block分5个module,第三个block分3个module
    # 每个block的module都是前一个module的重复,只是卷积核个数不一样,并且输入的输出是下一层的输入
    # ------------------------------------------------------------------------------------------------------------------
    # block1_module1
    branch1x1 = conv_bn(x,64,1,1)
    branch5x5 = conv_bn(x,48,1,1)
    branch5x5 = conv_bn(branch5x5,64,5,5)
    branch3x3 = conv_bn(x,64,1,1)
    branch3x3 = conv_bn(branch3x3,96,3,3)
    branch3x3 = conv_bn(branch3x3,96,3,3)
    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv_bn(branch_pool,32,1,1)
    x = layers.concatenate([branch1x1,branch5x5,branch3x3,branch_pool],axis=3,name='block1_module1')
    # block1_module2,重复module1,唯一区别是branch_pool卷积核数量变成了64
    branch1x1 = conv_bn(x, 64, 1, 1)
    branch5x5 = conv_bn(x, 48, 1, 1)
    branch5x5 = conv_bn(branch5x5, 64, 5, 5)
    branch3x3double = conv_bn(x, 64, 1, 1)
    branch3x3double = conv_bn(branch3x3double, 96, 3, 3)
    branch3x3double = conv_bn(branch3x3double, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3double, branch_pool], axis=3, name='block1_module2')
    # block1_module3,重复module2
    branch1x1 = conv_bn(x, 64, 1, 1)
    branch5x5 = conv_bn(x, 48, 1, 1)
    branch5x5 = conv_bn(branch5x5, 64, 5, 5)
    branch3x3double = conv_bn(x, 64, 1, 1)
    branch3x3double = conv_bn(branch3x3double, 96, 3, 3)
    branch3x3double = conv_bn(branch3x3double, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3double, branch_pool], axis=3, name='block1_module3')

    # ------------------------------------------------------------------------------------------------------------------
    # block2_module1,作为block的入口,要使得维度对齐,即288→768
    # 一个3*3,然后把一个5*5分解成了两个3*3,附带一个3*3池化
    # 然后还改了padding,导致计算方式发生变化
    branch3x3 = conv_bn(x,384,3,3,strides=(2,2),padding='valid')
    branch3x3double = conv_bn(x,64,1,1)
    branch3x3double = conv_bn(branch3x3double,96,3,3)
    branch3x3double = conv_bn(branch3x3double,96,3,3,strides=(2,2),padding='valid')
    branch_pool = MaxPooling2D((3,3),strides=(2,2),padding='valid')(x)
    x = layers.concatenate([branch3x3,branch3x3double,branch_pool],axis=3,name='block2_module1')
    # block2_module2,后面3个都是module2的重复
    branch1x1 = conv_bn(x,192,1,1)
    branch7x7 = conv_bn(x,128,1,1)
    branch7x7 = conv_bn(branch7x7,128,1,7)
    branch7x7 = conv_bn(branch7x7,192,7,1)
    branch7x7double = conv_bn(x,128,1,1)
    branch7x7double = conv_bn(branch7x7double,128,7,1)
    branch7x7double = conv_bn(branch7x7double,128,1,7)
    branch7x7double = conv_bn(branch7x7double, 128, 7, 1)
    branch7x7double = conv_bn(branch7x7double, 128, 1, 7)
    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv_bn(branch_pool,192,1,1)
    x = layers.concatenate([branch1x1,branch7x7,branch7x7double,branch_pool],axis=3,name='block2_module2')
    # block2_module3&&block2_module4
    for i in range(2):
        branch1x1 = conv_bn(x, 192, 1, 1)
        branch7x7 = conv_bn(x, 160, 1, 1)
        branch7x7 = conv_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv_bn(branch7x7, 192, 7, 1)
        branch7x7double = conv_bn(x, 160, 1, 1)
        branch7x7double = conv_bn(branch7x7double, 160, 7, 1)
        branch7x7double = conv_bn(branch7x7double, 160, 1, 7)
        branch7x7double = conv_bn(branch7x7double, 160, 7, 1)
        branch7x7double = conv_bn(branch7x7double, 192, 1, 7)
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7double, branch_pool], axis=3, name='block2_module'+str(2+i))
    # block3_module5
    branch1x1 = conv_bn(x, 192, 1, 1)
    branch7x7 = conv_bn(x, 192, 1, 1)
    branch7x7 = conv_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv_bn(branch7x7, 192, 7, 1)
    branch7x7double = conv_bn(x, 192, 1, 1)
    branch7x7double = conv_bn(branch7x7double, 192, 7, 1)
    branch7x7double = conv_bn(branch7x7double, 192, 1, 7)
    branch7x7double = conv_bn(branch7x7double, 192, 7, 1)
    branch7x7double = conv_bn(branch7x7double, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), 1, 1, strides=(1, 1), padding='same')(x)
    branch_pool = conv_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7double, branch_pool], axis=3, name='block2_module5')

    # ------------------------------------------------------------------------------------------------------------------
    # block3_module1
    branch3x3 = conv_bn(x,192,1,1)
    branch3x3 = conv_bn(branch3x3,320,3,3,strides=(2,2),padding='valid')
    branch7x7x3 = conv_bn(x,192,1,1)
    branch7x7x3 = conv_bn(branch7x7x3,192,1,7)
    branch7x7x3 = conv_bn(branch7x7x3,192,7,1)
    branch7x7x3 = conv_bn(branch7x7x3,192,3,3,strides=(2,2),padding='valid')
    branch_pool = MaxPooling2D((3,3),strides=(2,2),padding='valid')(x)
    x = layers.concatenate(branch3x3,branch7x7x3,branch_pool,axis=3,name='block3_module1')
    # block3_module2 && block3_module3
    for j in range(2):
        branch1x1 = conv_bn(x, 320, 1, 1)
        branch3x3 = conv_bn(x,384,1,1)
        branch3x3_1 = conv_bn(branch3x3,384,1,7)
        branch3x3_2 = conv_bn(branch3x3,384,7,1)
        branch3x3 = layers.concatenate([branch3x3_1,branch3x3_2],axis=3,name='block3_module_'+str(2+j))
        branch3x3double = conv_bn(x,448,1,1)
        branch3x3double = conv_bn(branch3x3double,384,1,1)
        branch3x3double_1 = conv_bn(branch3x3double,384,1,3)
        branch3x3double_2 = conv_bn(branch3x3double,384,3,1)
        branch3x3double = layers.concatenate([branch3x3double_1,branch3x3double_2],axis=3)
        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv_bn(branch_pool,192,1,1)
        x = layers.concatenate([branch1x1,branch3x3,branch3x3double,branch_pool],axis=3,name='block3_module'+str(2+j))
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(classes,activation='softmax',name='prediction')(x)

    inputs = input_shape
    model = my_InceptionV3(inputs,x,name='myInceptionV3')
    return model

def preprocessInput(x):
    x/=255.
    x-=0.5
    x*=2.
    return x

if __name__ == '__main__':
    model = my_InceptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    img_path = "elephant.jpg"
    img = image.load_img(img_path,target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)

    x = preprocessInput(x)
    predict = model.predict(x)
    print('predicted:',decode_predictions(predict))














