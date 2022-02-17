"""
基于DNN的mnist手写数字识别，采用tensorFlow.keras框架实现
"""
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# 1.加载mnist数据集
(src_train_images, src_train_labels), (src_test_images, src_test_labels) = keras.datasets.mnist.load_data()
print("src_train_images.shape = {}, src_train_labels = {}".format(src_train_images.shape, src_train_labels.shape))
print("src_test_images.shape = {}, src_test_labels = {}".format(src_test_images.shape, src_test_labels.shape))
print(src_train_labels[0:10])

# 2.数据预处理
# 图像归一化
train_images = src_train_images / 255.0
test_images = src_test_images / 255.0

# label转换为one-hot编码
train_labels = keras.utils.to_categorical(src_train_labels)
test_labels = keras.utils.to_categorical(src_test_labels)

# 3.构建网络
DNN = keras.models.Sequential()
DNN.add(keras.layers.Flatten(input_shape=(28, 28)))
DNN.add(keras.layers.Dense(128, activation='relu'))
DNN.add(keras.layers.Dropout(0.5))
DNN.add(keras.layers.Dense(128, activation='relu'))
DNN.add(keras.layers.Dropout(0.5))
DNN.add(keras.layers.Dense(10, activation='softmax'))
DNN.summary()

# 4.编译网络
DNN.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
            loss=keras.losses.categorical_crossentropy,
            metrics=['acc'])

# 5.训练网络
history = DNN.fit(train_images, train_labels, batch_size=128, epochs=50,
                  validation_data=(test_images, test_labels))

# 6.查看精度曲线和损失曲线
keys = history.history.keys()
# print(keys)     # 结果：dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
plt.figure()
plt.plot(history.epoch, history.history.get('acc'), label='train_acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()

plt.figure()
plt.plot(history.epoch, history.history.get('loss'), label='train_loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()

# 7.网络性能评估
test_loss, test_acc = DNN.evaluate(test_images, test_labels, batch_size=128)
print('Evaluate：test_acc = {}, test_loss = {}'.format(test_acc, test_loss))

# 8.随机选取部分样本进行识别预测
idxs = np.random.randint(0, 10000, size=10)
test_sample_images = test_images[idxs]
test_sample_labels = test_labels[idxs]
print('Predict: True labels = ', [np.argmax(label) for label in test_sample_labels])

predict_results = DNN.predict(test_sample_images)
# 将概率值转换为类别
predict_classes = [np.argmax(result) for result in predict_results]
print('Predict: Predict labels = ', predict_classes)
