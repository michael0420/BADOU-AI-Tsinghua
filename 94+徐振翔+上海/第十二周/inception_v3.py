import numpy as np
from tensorflow.python.keras import Input, layers, Model
from tensorflow.python.keras.applications.densenet import decode_predictions
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D, Dense
from tensorflow.python.keras.preprocessing import image


def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    '''
    Con2d + BN+ Relu
    :param x:输入模型
    :param filters:输出特征图数量（卷积核个数）
    :param num_row:卷积核行数
    :param num_col:卷积核列数
    :param strides:步长
    :param padding:填充方式
    :param name:卷积核名称
    :return:Conv2d + BatchNormalization + Relu 后的模型结果
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    # 卷积
    x = Conv2D(filters=filters, kernel_size=(num_row, num_col), strides=strides, padding=padding, use_bias=False,
               name=conv_name)(x)
    # BN,scale为True时，添加缩放因子gamma到该BN层，否则不添加。添加gamma是对BN层的变化加入缩放操作。注意，gamma一般设定为可训练参数，即trainable = True。
    x = BatchNormalization(scale=False, name=bn_name)(x)
    # Relu
    x = Activation(activation='relu', name=name)(x)
    return x


def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    '''
    Inception V3 网络
    :param input_shape: 输入尺寸
    :param classes: 分类结果数量
    :return: 模型
    '''
    # 1、调整输入图像格式为299*299*3
    img_input = Input(shape=input_shape)

    # 2、base层，3次卷积， 1次最大池化， 3次卷积 得到Inception结构的输入base
    # 采用3*3卷积核，步长为2*2，输出特征为32，padding方式为valid，输出尺寸 149*149*32
    x = conv2d_bn(img_input, 32, 3, 3, (2, 2), 'valid')
    # 采用3*3卷积核，步长为1*1，输出特征为32，padding方式为valid，输出尺寸 147*147*32
    x = conv2d_bn(x, 32, 3, 3, (1, 1), 'valid')
    # 采用3*3卷积核，步长为1*1，输出特征为64，padding方式为same，输出尺寸 147*147*64
    x = conv2d_bn(x, 64, 3, 3, (1, 1), 'same')

    # 最大池化，采用3*3卷积核，步长为2*2，输出尺寸73*73*64
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # -----------------------------------------------------------------------#
    # 此处与标准模式有区别，不确定原因，ppt中用的是3个卷积，参考代码中用的是2个卷积一个maxpooling，且有一个卷积与描述存在差异 !!!!!!
    # 为了保持权重参数能正常使用，保持原代码模型结构
    # -----------------------------------------------------------------------#
    # 论文中及ppt中做法如下：
    # 采用3*3卷积核，步长为1*1，输出特征为80，padding方式为valid，输出尺寸 71*71*80
    # x = conv2d_bn(x, 80, 3, 3, (1, 1), 'valid')
    # # 采用3*3卷积核，步长为2*2，输出特征为192，padding方式为valid，输出尺寸 35*35*192
    # x = conv2d_bn(x, 192, 3, 3, (2, 2), 'valid')
    # # 采用3*3卷积核，步长为1*1，输出特征为288，padding方式为same，输出尺寸 35*35*288
    # x = conv2d_bn(x, 288, 3, 3, (1, 1), 'same')
    # print(x.shape)
    # 参考代码中做法：
    # 计划：采用3*3卷积核，步长为1*1，输出特征为80，padding方式为valid，输出尺寸 71*71*80
    # 实际：采用1*1卷积核，步长为1*1，输出特征为80，padding方式为valid，输出尺寸 73*73*80
    x = conv2d_bn(x, 192, 1, 1, (1, 1), 'valid')
    # 实际：采用3*3卷积核，步长为1*1，输出特征为192，padding方式为valid，输出尺寸 71*71*192
    x = conv2d_bn(x, 192, 3, 3, (1, 1), 'valid')
    # 实际：最大池化，采用3*3卷积核，步长为2*2，输出尺寸 35*35*192
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 为了保持权重参数可用，以下均为参考代码中的结构而不一定是论文中的标准结构

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35*35*192->35*35*256
    # 1*1卷积分支，same填充,64特征
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 5*5卷积分支，same填充,64特征(V4中，5*5卷积使用2个3*3代替)
    # 先进行1*1卷积，再进行5*5卷积
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 3*3卷积分支，same填充,96特征
    # 先进行1*1卷积，再进行两次3*3卷积
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    # 均值池化分支 32特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same，再进行1*1卷积
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    # 最终对特征拼接得到 64+64+96+32 = 256
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed0')

    # --------------------------------#
    # Block1 part 2
    # 35*35*256->35*35*288
    # 1*1卷积分支，same填充,64特征
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 5*5卷积分支，same填充,64特征
    # 先进行1*1卷积，再进行5*5卷积
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 3*3卷积分支，same填充,96特征
    # 先进行1*1卷积，再进行两次3*3卷积
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    # 均值池化分支 32特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same，再进行1*1卷积
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 最终对特征拼接得到 64+64+96+64 = 288
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed1')

    # --------------------------------#
    # Block1 part 3
    # 35*35*288->35*35*288
    # 1*1卷积分支，same填充,64特征
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    # 5*5卷积分支，same填充,64特征
    # 先进行1*1卷积，再进行5*5卷积
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    # 3*3卷积分支，same填充,96特征
    # 先进行1*1卷积，再进行两次3*3卷积
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    # 均值池化分支 32特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same，再进行1*1卷积
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 最终对特征拼接得到 64+64+96+64 = 288
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35*35*288->17*17*768
    # 单次3*3卷积分支，384特征
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    # 两次3*3卷积分支，same填充,96特征
    # 先进行1*1卷积，再进行两次3*3卷积
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')
    # 最大池化分支 288特征
    # 进行max_pooling,3*3卷积核，步长2*2
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    # 最终对特征拼接得到 384+96+288 = 768
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # --------------------------------#
    # Block2 part2
    # 17*17*768->17*17*768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 单次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行一次（1*7卷积+7*1卷积）
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行两次（1*7卷积+7*1卷积）
    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # 均值池化分支 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same，再进行1*1卷积
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 最终对特征拼接得到 192+192+192+192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    # --------------------------------#
    # Block2 part3
    # 17*17*768->17*17*768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 单次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行一次（1*7卷积+7*1卷积）
    branch7x7 = conv2d_bn(x, 160, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行两次（1*7卷积+7*1卷积）
    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # 均值池化分支 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same，再进行1*1卷积
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 最终对特征拼接得到 192+192+192+192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed5')

    # --------------------------------#
    # Block2 part4
    # 17*17*768->17*17*768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 单次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行一次（1*7卷积+7*1卷积）
    branch7x7 = conv2d_bn(x, 160, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行两次（1*7卷积+7*1卷积）
    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # 均值池化分支 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same，再进行1*1卷积
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 最终对特征拼接得到 192+192+192+192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed6')

    # --------------------------------#
    # Block2 part5
    # 17 * 17 * 768->17 * 17 * 768
    # 1*1卷积分支，same填充,192特征
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    # 单次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行一次（1*7卷积+7*1卷积）
    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    # 两次7*7卷积分支，same填充,192特征
    # 先进行1*1卷积，再进行两次（1*7卷积+7*1卷积）
    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    # 均值池化分支 192特征
    # 先进行avg_pooling,3*3卷积核，步长1*1，填充same，再进行1*1卷积
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 最终对特征拼接得到 192+192+192+192 = 768
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17*17*768->8*8*1280
    # 3*3卷积分支，valid填充,步长为2*2，320特征
    # 先进行1*1卷积，再进行一次3*3卷积
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    # 7*7卷积分支，same与valid填充,192特征
    # 先进行1*1卷积，再进行一次（1*7卷积+7*1卷积），最后进行一次3*3卷积
    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    # 最大池化分支 768特征
    # 进行max_pooling,3*3卷积核，步长2*2
    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # 最终对特征拼接得到 320+192+768 = 1280
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # --------------------------------#
    # Block3 part2 和 part3
    # 8*8*1280->8*8*1280
    for i in range(2):
        # 1*1卷积分支，same填充,320特征
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        # 单次3*3卷积分支，输出768特征
        # 先进行1*1卷积，再分别进行一次1*3与一次3*1卷积并最终拼接
        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        # 两次3*3卷积分支，输出768特征
        # 先进行1*1卷积，在进行一次3*3卷积，然后分别进行一次1*3与一次3*1卷积并最终拼接
        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        # 均值池化分支 192特征
        # 先进行均值池化,3*3卷积核，步长1*1，填充same，再进行1*1卷积
        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # 最终对特征拼接得到 320+768+768+192 = 2048
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))

    # 最终，平均池化后全连接。
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # 构造模型并返回
    inputs = img_input
    model = Model(inputs, x, name='inception_v3')

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
