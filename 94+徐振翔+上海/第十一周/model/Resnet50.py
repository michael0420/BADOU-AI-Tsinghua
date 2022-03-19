from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras import layers, Input, Model
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, Dropout, BatchNormalization, Flatten, \
    Activation, Dense


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    分支中有卷积的Block
    :param input_tensor: 输入tensor
    :param kernel_size: 核大小
    :param filters: 各卷积filter
    :param stage: 阶段号，用于显示层对应信息，不影响实际运行
    :param block: 阶段中的第几个block
    :param strides: 步长
    :return: conv_block模型
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 左分支，正常卷积
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 右分支，单步执行
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    :param input_tensor: 输入tensor
    :param kernel_size:卷积核大小
    :param filters:各卷积filter
    :param stage: 阶段号，用于显示层对应信息，不影响实际运行
    :param block: 阶段中的第几个block
    :return: identity_block模型
    """
    filters1, filters2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 求和
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


# def ResNet50(input_shape=[224, 224, 3], classes=1000):
def ResNet50(input_shape=[224, 224, 3], classes=2):
    # ResNet50 网络本体
    # Input:把tensor实例化
    img_input = Input(shape=input_shape)
    # 行列padding
    x = layers.ZeroPadding2D((3, 3))(img_input)

    # 初始特征提取：卷积、BN、激活、最大池化
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 第一阶段，1个conv Block + 2个Identity Block
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 第二阶段，1个conv Block + 3个Identity Block
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 第三阶段，1个conv Block + 5个Identity Block
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 第四阶段，1个conv Block + 2个Identity Block
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 均值池化
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # 拍扁
    x = Flatten()(x)
    # 全连接
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return model
