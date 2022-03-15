from keras import models
from keras import layers
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
"""
Conv2d + BatchNormalization
输入：
x 输入数据，卷积+BN+激活后的结果
filters 输出特征数
num_row 卷积核行数
num_col 卷积核列数
strides 步长,默认(1,1)
padding 填充方式，默认same
name 卷积核名
"""
def conv2d_bn(x,filters,num_row,num_col,strides=(1,1),padding='same',name=None):
    if name is not None:
        bn_name = name+'_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    #卷积
    x = layers.Conv2D(filters=filters,kernel_size=(num_row,num_col),strides=strides,padding=padding,use_bias=False,name=conv_name)(x)
    #BatchNormalization
    x = layers.BatchNormalization(scale=False,name=bn_name)(x)
    x = layers.Activation('relu',name=name)(x)
    return x

"""
Inception V3网络
"""
def InceptionV3(input_shape=[299,299,3],classes=1000):
    img_input = layers.Input(shape=input_shape,)

    # 采用3*3卷积核，步长为2*2，输出特征为32，padding方式为valid
    x = conv2d_bn(img_input,32,3,3,(2,2),'valid')
    # 采用3*3卷积核，步长为1*1，输出特征为32，padding方式为valid
    x = conv2d_bn(x,32,3,3,(1,1),'valid')
    # 采用3*3卷积核，步长为1*1，输出特征为64，padding方式为valid
    x = conv2d_bn(x, 64, 3, 3, (1, 1), 'valid')
    #max_pooling
    x= layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(x)

    # 采用3*3卷积核，步长为1*1，输出特征为80，padding方式为valid
    x = conv2d_bn(x, 80, 1, 1, (1, 1), 'valid')
    # 采用3*3卷积核，步长为1*1，输出特征为192，padding方式为valid
    x = conv2d_bn(x, 192, 3, 3, (1, 1), 'valid')
    # max_pooling
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    #----------------Block1 part1--------------------
    #part1 35*35*192->35*35*256
    #1*1卷积分支，same填充,64特征
    branch1x1 = conv2d_bn(x,64,1,1)

    #5*5卷积分支，same填充,64特征
    #先进行1*1卷积
    branch5x5 = conv2d_bn(x,48,1,1)
    #再进行5*5卷积
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    #3*3卷积分支，same填充,96特征
    #先进行1*1卷积
    branch3x3 = conv2d_bn(x,64,1,1)
    #再进行两次3*3卷积
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    #avg_pooling 32特征
    #先进行avg_pooling,3*3卷积核，步长1*1，填充same
    branch_pooling = layers.AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    #再进行1*1卷积
    branch_pooling = conv2d_bn(branch_pooling,32,1,1)

    #基于特征层concatenate 64+64+96+32=256,
    x = layers.concatenate([branch1x1,branch5x5,branch3x3,branch_pooling],axis=3,name='mixed0')

    #---------------------Block1 part1------------------------------

    # ----------------Block1 part2--------------------
    # part2 35*35*256->35*35*288
    # 1*1卷积分支，same填充,64特征
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 5*5卷积分支，same填充,64特征
    # 先进行1*1卷积
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    # 再进行5*5卷积
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 3*3卷积分支，same填充,96特征
    # 先进行1*1卷积
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    # 再进行两次3*3卷积
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    # avg_pooling 32特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same
    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 再进行1*1卷积
    branch_pooling = conv2d_bn(branch_pooling, 64, 1, 1)

    # 基于特征层concatenate 64+64+96+64=288,
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pooling],axis=3,name='mixed1')

    # ---------------------Block1 part2------------------------------

    # ----------------Block1 part3--------------------
    # part2 35*35*288->35*35*288
    # 1*1卷积分支，same填充,64特征
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 5*5卷积分支，same填充,64特征
    # 先进行1*1卷积
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    # 再进行5*5卷积
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 3*3卷积分支，same填充,96特征
    # 先进行1*1卷积
    branch3x3 = conv2d_bn(x, 64, 1, 1)
    # 再进行两次3*3卷积
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    # avg_pooling 32特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same
    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 再进行1*1卷积
    branch_pooling = conv2d_bn(branch_pooling, 64, 1, 1)

    # 基于特征层concatenate 64+64+96+64=288,
    x = layers.concatenate([branch1x1, branch5x5, branch3x3, branch_pooling], axis=3, name='mixed2')

    # ---------------------Block1 part3------------------------------

    #----------------Block2 part1--------------------
    #part1 35*35*288->17*17*768
    #单3*3卷积分支
    branch3x3 = conv2d_bn(x, 384, 3, 3,strides=(2,2),padding='valid')
    # 3*3卷积分支，same填充,96特征
    # 先进行1*1卷积
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    # 再进行两次3*3卷积
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,strides=(2,2),padding='valid')
    #Max_pooling
    #进行max_pooling,3*3卷积核，步长2*2
    branch_pooling = layers.MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    #基于特征层concatenate 384+96+288=768,
    x = layers.concatenate([branch3x3,branch3x3dbl,branch_pooling],axis=3,name='mixed3')
    #---------------------Block2 part1------------------------------

    # ----------------Block2 part2--------------------
    # part2 17*17*768->17*17*768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    # 再进行1*7卷积
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    # 再进行7*1卷积
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # avg_pooling 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same
    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 再进行1*1卷积
    branch_pooling = conv2d_bn(branch_pooling, 192, 1, 1)

    # 基于特征层concatenate 192+192+192+192=768,
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pooling], axis=3, name='mixed4')

    # ---------------------Block2 part2------------------------------

    # ----------------Block2 part3--------------------
    # part2 17*17*768->17*17*768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7 = conv2d_bn(x, 160, 1, 1)
    # 再进行1*7卷积
    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
    # 再进行7*1卷积
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # avg_pooling 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same
    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 再进行1*1卷积
    branch_pooling = conv2d_bn(branch_pooling, 192, 1, 1)

    # 基于特征层concatenate 192+192+192+192=768,
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pooling], axis=3, name='mixed5')

    # ---------------------Block2 part3------------------------------

    # ----------------Block2 part4--------------------
    # part2 17*17*768->17*17*768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7 = conv2d_bn(x, 160, 1, 1)
    # 再进行1*7卷积
    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
    # 再进行7*1卷积
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # avg_pooling 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same
    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 再进行1*1卷积
    branch_pooling = conv2d_bn(branch_pooling, 192, 1, 1)

    # 基于特征层concatenate 192+192+192+192=768,
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pooling], axis=3, name='mixed6')

    # ---------------------Block2 part4------------------------------

    # ----------------Block2 part5--------------------
    # part2 17*17*768->17*17*768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7 = conv2d_bn(x, 192, 1, 1)
    # 再进行1*7卷积
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    # 再进行7*1卷积
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    # 再进行1*7卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    # 再进行7*1卷积
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # avg_pooling 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same
    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    # 再进行1*1卷积
    branch_pooling = conv2d_bn(branch_pooling, 192, 1, 1)

    # 基于特征层concatenate 192+192+192+192=768,
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pooling], axis=3, name='mixed7')

    # ---------------------Block2 part5------------------------------
    # ----------------Block3 part1--------------------
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2,2), padding='valid')

    # 7*7卷积分支，same填充,192特征
    # 先进行1*1卷积
    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    # 再进行1*7卷积
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    # 再进行7*1卷积
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    # 再进行3*3卷积
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    # max_pooling 192特征
    # max_pooling,3*3卷积核，步长2*2，填充same
    branch_pooling = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 基于特征层concatenate 64+64+96+64=288,
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pooling], axis=3, name='mixed8')

    # ---------------------Block3 part1------------------------------

    # ----------------Block3 part2--------------------
    # 8 x 8 x 1280 -> 8 x 8 x 2048
    # 1*1卷积分支，same填充,320特征
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_1')

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1),padding='same')(x)
    branch_pooling = conv2d_bn(branch_pooling,192,1,1)
    # 基于特征层concatenate 320+384*2+384*2+192 = 2048
    x = layers.concatenate([branch1x1, branch3x3,branch3x3dbl,branch_pooling], axis=3, name='mixed9')

    # ---------------------Block3 part2------------------------------

    # ----------------Block3 part3--------------------
    # 8 x 8 x 2048 -> 8 x 8 x 2048
    # 1*1卷积分支，same填充,320特征
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_2')

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

    branch_pooling = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pooling = conv2d_bn(branch_pooling, 192, 1, 1)
    # 基于特征层concatenate 320+384*2+384*2+192 = 2048
    x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pooling], axis=3, name='mixed10')

    # ---------------------Block3 part3------------------------------

    # 平均池化后全连接。
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    model = models.Model(inputs, x, name='inception_v3')

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = InceptionV3()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
