"""
基于tensorflow.keras的ResNet分类网络实现
tensorflow: 2.3.0
"""
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import DepthwiseConv2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


def relu6(x):
    return keras.backend.relu(x, max_value=6)


def conv_bn_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, strides, padding, use_bias=False, name='conv_0')(inputs)
    x = BatchNormalization(name='bn_0')(x)
    x = Activation(relu6, name='relu_0')(x)
    return x


def depthwise_conv_block(inputs, depthwise_strides, pointwise_filters, block_id=1):
    # depthwise conv block
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=depthwise_strides,
                        padding='same',
                        depth_multiplier=1,
                        use_bias=False,
                        name='conv_dw_' + str(block_id))(inputs)
    x = BatchNormalization(name='conv_dw_bn_' + str(block_id))(x)
    x = Activation(relu6, name='conv_dw_relu_' + str(block_id))(x)
    # pointwise conv block
    x = Conv2D(filters=pointwise_filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               name='conv_pw_' + str(block_id))(x)
    x = BatchNormalization(name='conv_pw_bn_' + str(block_id))(x)
    x = Activation(relu6, name='conv_pw_relu_'+ str(block_id))(x)
    return x


def MobileNetV1(input_shape=(224, 224, 3), dropout_rate=0.5, num_classes=1000):
    input_tensor = Input(input_shape)

    # 224*224*3 -> 112*112*32
    x = conv_bn_block(input_tensor, 32, (3, 3), (2, 2), 'same')

    # 112*112*32 -> 112*112*64
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=64, block_id=1)

    # 112*112*64 -> 56*56*128
    x = depthwise_conv_block(x, depthwise_strides=(2, 2), pointwise_filters=128, block_id=2)

    # 56*56*128 -> 56*56*128
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=128, block_id=3)

    # 56*56*128 -> 28*28*256
    x = depthwise_conv_block(x, depthwise_strides=(2, 2), pointwise_filters=256, block_id=4)

    # 28*28*256 -> 28*28*256
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=256, block_id=5)

    # 28*28*256 -> 14*14*512
    x = depthwise_conv_block(x, depthwise_strides=(2, 2), pointwise_filters=512, block_id=6)

    # （5个） 14*14*512 -> 14*14*512
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=512, block_id=7)
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=512, block_id=8)
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=512, block_id=9)
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=512, block_id=10)
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=512, block_id=11)

    # 14*14*512 -> 7*7*1024
    x = depthwise_conv_block(x, depthwise_strides=(2, 2), pointwise_filters=1024, block_id=12)

    # 7*7*1024 -> 7*7*1024
    x = depthwise_conv_block(x, depthwise_strides=(1, 1), pointwise_filters=1024, block_id=13)

    # 7*7*1024 -> 1*1*1024
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    # 1*1*1024 -> 1*1*num_classes
    x = Dropout(rate=dropout_rate, name='dropout')(x)
    x = Dense(num_classes, name='fc')(x)

    model = keras.Model(input_tensor, x, name='MobileNet_V1')
    return model

