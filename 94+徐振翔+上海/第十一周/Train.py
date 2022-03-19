import cv2
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution() #关闭eager运算
# tf.disable_v2_behavior() #禁用TensorFlow 2.x行为
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
import utils
from model.AlexNet import AlexNet
from model.Vgg16 import Vgg16Net
from model.Resnet50 import ResNet50


def generator_data(lines, batch_size):
    """
    制作一个图像数据生成器
    :param lines:
    :param batch_size:输出batch大小
    :return: imgs:训练用图像集。labels:图像对应的标签集
    """
    n = len(lines)
    i = 0
    while True:
        imgs = []
        labels = []
        for batch in range(batch_size):
            if i == 0:
                # 打乱图片的序号
                np.random.shuffle(lines)
            ll = lines[i].split(';')
            # 获取图片名称及标签
            name = ll[0]
            label = ll[1]
            # 从文件中读取图像
            img = cv2.imread(utils.base_path + './data/image/train' + '/' + name)
            # 归一化 (h,w,c)
            img_nor = img / 255
            # 保存图片与标签到列表
            imgs.append(img_nor)
            labels.append(label)
            # 迭代器如果遍历了所有图片，则重新开始
            i = (i + 1) % n
        # 图像预处理
        # 调整图像尺寸为 batch_size*224*224*3 (n,h,w,c)
        img_resize = utils.resize(imgs, (224, 224))
        # img_resize = np.resize(img_nor, (224, 224, 3))
        # print(img_resize.shape)
        # img_resize = cv2.resize(img_nor, (224, 224))
        # print(img_resize.shape)
        # 转换图像维度从(n,h,w,c)变为(n,c,h,w) (1, 3, 224, 224)
        # img_resize = img_resize.transpose(0, 3, 1, 2)
        # 讲标签转换为one-hot编码(num_classes:种类数)
        labels = to_categorical(labels, num_classes=2)
        yield img_resize, labels


if __name__ == '__main__':
    # 模型权重参数保存位置
    log_dir = utils.base_path + "/logs/"
    # 加载数据集txt
    with open(utils.base_path + "/data/dataset.txt", "r") as f:
        lines = f.readlines()
    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    # 使用相同的随机数seed可以保证生成的lines第一轮是一样的
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 样本总数
    total_num = len(lines)
    # 90%用于训练集，10%用于测试集
    train_num = int(total_num * 0.9)
    evaluate_num = total_num - train_num

    # 实例化模型
    if utils.net == 1:
        model = AlexNet()
    elif utils.net == 2:
        model = Vgg16Net()
    elif utils.net == 3:
        model = ResNet50()
    else:
        model = AlexNet()

    # 保存的方式，3代保存一次(记录模型检查点)
    # 该回调函数将在每个epoch后保存模型信息到filepath
    # keras.callbacks.ModelCheckpoint(filepath,
    #                                 monitor='val_loss',
    #                                 verbose=0,
    #                                 save_best_only=False,
    #                                 save_weights_only=False,
    #                                 mode='auto',
    #                                 period=1)
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='accuracy',
                                        save_weights_only=False,
                                        save_best_only=True,
                                        period=3
                                        )
    # 当监测值不再改善时，该回调函数将中止训练
    # 当val_loss不再下降时，可以早停
    # keras.callbacks.EarlyStopping(monitor='val_loss',
    #                               patience=0,
    #                               verbose=0,
    #                               mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=10,
                                   verbose=1
                                   )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    # monitor：监测的值，可以是accuracy，val_loss, val_accuracy
    # factor：缩放学习率的值，学习率将以lr = lr * factor的形式被减少
    # patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    # mode：‘auto’，‘min’，‘max’之一
    # 默认‘auto’就行
    # epsilon：阈值，用来确定是否进入检测值的“平原区”
    # cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    # min_lr：学习率最小值，能缩小到的下限
    # verbose:触发条件后print
    reduce_lr = ReduceLROnPlateau(monitor='accuracy',
                                  factor=0.5,
                                  patience=3,
                                  verbose=1
                                  )

    # 设置损失函数，优化器函数，评价函数，交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy']
                  )
    # 批处理
    batch_size = 32

    print("Train on {} samples, evaluate on {} samples, with batch size {}."
          .format(train_num, evaluate_num, batch_size))

    # 开始训练
    # generator: 生成器函数，输出应该是形为（inputs, target）或者（inputs, targets, sample_weight）的元组，生成器会在数据集上无限循环
    # steps_per_epoch: 顾名思义，每轮的步数，整数，当生成器返回steps_per_epoch次数据时，进入下一轮。
    # epochs: 整数，数据的迭代次数
    # verbose：日志显示开关。0:代表不输出日志，1:代表输出进度条记录，2:代表每轮输出一行记录
    # validation_data：验证集数据.
    # max_queue_size: 整数.迭代器最大队列数，默认为10
    # workers: 最大进程数。在使用多线程时，启动进程最大数量（process - based threading）。未特别指定时，默认为1。如果指定为0，则执行主线程.
    # use_multiprocessing: 布尔值。True: 使用基于过程的线程
    model.fit_generator(generator=generator_data(lines[:train_num], batch_size),
                        steps_per_epoch=max(1, int(train_num // batch_size)),
                        # epochs=50,
                        epochs=1,
                        verbose=1,
                        validation_data=generator_data(lines[:train_num], batch_size),
                        validation_steps=max(1, evaluate_num // batch_size),
                        initial_epoch=0,
                        callbacks=[checkpoint_period, reduce_lr]
                        )
    # 保存模型权重参数
    model.save_weights(log_dir + 'last1.h5')
