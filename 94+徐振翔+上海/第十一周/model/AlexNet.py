"""
AlexNet 网络结构:
池化核大小预设为3*3
1、一张原始图片被resize到(224,224,3);
2、使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96);
3、使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
4、使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256);
5、使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)
6、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384);
7、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384);
8、使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256);
9、使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256);
10、两个全连接层，最后输出为1000类(先拍扁)
"""
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    """
    :param input_shape: 输入图像shape
    :param output_shape: 输出类别数
    :return:model:模型
    """
    # 构建AlexNet模型
    model = Sequential()
    # 2、输出的shape为(55,55,96), 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                     input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # 3、输出的shape为(27,27,96)， 使用步长为2的最大池化层进行池化
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # 4、输出的shape为(27,27,256)，使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    # 5、输出的shape为(13,13,256)，使用步长为2的最大池化层进行池化
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # 6、输出的shape为(13,13,384)，使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    # 7、输出的shape为(13,13,384)，使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    # 8、输出的shape为(13,13,256)，使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    # 9、输出的shape为(6,6,256)，使用步长为2的最大池化层进行池化
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    # 10、两个全连接层，最后输出为1000类(先拍扁)(当前项目为2类)
    # 数据拍扁
    model.add(Flatten())
    # 全连接层
    model.add(Dense(1024, activation='relu'))
    # 防止过拟合，使用dropout随机舍弃部分节点
    model.add(Dropout(0.25))
    # 第二层全连接
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    # 最终输出为两类，以适配猫狗分类
    model.add(Dense(output_shape, activation='softmax'))
    return model
