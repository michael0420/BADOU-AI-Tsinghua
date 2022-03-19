import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.densenet import decode_predictions
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, \
    GlobalAveragePooling2D, Reshape, \
    Dropout
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing import image


def relu6(x):
    """
    上限为6的relu
    """
    return K.relu(x, max_value=6)


def conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """
    卷积 + BN + relu6 激活 的 卷积 bolck
    :param inputs: 输入tensor
    :param filters: 卷积核数
    :param kernel: 卷积核大小
    :param strides: 步长
    :return: 模型结果
    """
    x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    """
    深度可分离卷积
    :param inputs: 输入tensor
    :param pointwise_conv_filters: 卷积核数
    :param depth_multiplier: 深度
    :param strides: 步长
    :param block_id: 块id
    :return: 模型
    """
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides,
                        use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def MobileNet(input_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    """
    MobilNet，适用于嵌入式的深度可分离网络
    :param input_shape: 输入tensor尺寸
    :param depth_multiplier: 深度数量
    :param dropout: dropout比例
    :param classes: 可分类数
    :return: 模型
    """
    img_input = Input(shape=input_shape)
    # 224,224,3 -> 112,112,32
    x = conv_block(img_input, 32, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 112,112,64 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    # 56,56,128 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    # 28,28,256 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    # 28,28,256 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    # 14,14,512 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=11)
    # 14,14,512 -> 7,7,1024
    x = depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)
    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    # 准备输出模型
    inputs = img_input
    # 与权重参数保存文件保持一致命名
    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1
