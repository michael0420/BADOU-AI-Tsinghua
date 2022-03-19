# VGG16的结构：
# 1、一张原始图片被resize到(224,224,3)。
# 2、conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)，再2X2最大池化，输出net为(112,112,64)。
# 3、conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)，再2X2最大池化，输出net为(56,56,128)。
# 4、conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)，再2X2最大池化，输出net为(28,28,256)。
# 5、conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(28,28,512)，再2X2最大池化，输出net为(14,14,512)。
# 6、conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(14,14,512)，再2X2最大池化，输出net为(7,7,512)。
# 7、利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)。共进行两次。
# 8、利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)。
# 最后输出的就是每个类的预测。

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense


def Vgg16Net(input_shape=(224, 224, 3), output_shape=2):
    """
    :param input_shape: 输入图像shape
    :param output_shape: 输出类别数
    :return:model:模型
    """
    # 构建AlexNet模型
    model = Sequential()

    # 2、conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)，再2X2最大池化，输出net为(112,112,64)。
    # (224,224,3)->(224,224,64)
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (224,224,64)->(224,224,64)
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (224,224,64)->(112,112,64)
    # pool_size = (2, 2), 池化核的尺寸，默认是2×2
    # strides = None，移动步长的意思 ，默认是池化核尺寸，即2，所以这里可以不写
    # padding = ‘valid’, 是否填充，，默认是不填充
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3、conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)，再2X2最大池化，输出net为(56,56,128)。
    # (112,112,64)->(112,112,128)
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (112,112,128)->(112,112,128)
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (112,112,128)->(56,56,128)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 4、conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)，再2X2最大池化，输出net为(28,28,256)。
    # (56,56,128)->(56,56,256)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (56,56,256)->(56,56,256)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (56,56,256)->(56,56,256)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (56,56,256)->(28,28,256)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 5、conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(28,28,512)，再2X2最大池化，输出net为(14,14,512)。
    # (28,28,256)->(28,28,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (28,28,512)->(28,28,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (28,28,512)->(28,28,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (28,28,512)->(14,14,512)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 6、conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(14,14,512)，再2X2最大池化，输出net为(7,7,512)。
    # (14,14,512)->(14,14,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (14,14,512)->(14,14,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (14,14,512)->(14,14,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 默认添加的BN层
    # (14,14,512)->(7,7,512)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 7、利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)。共进行两次。
    # (7,7,512)->(1,1,4096),注意：卷积核为7*7
    model.add(Conv2D(filters=4096, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())  # 默认添加的BN层
    # (1,1,4096)->(1,1,4096),注意：卷积核为1*1
    model.add(Conv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.5))

    # 8、利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)。
    model.add(Conv2D(filters=output_shape, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='softmax'))

    # 最终输出前需要拍扁
    model.add(Flatten())

    return model
