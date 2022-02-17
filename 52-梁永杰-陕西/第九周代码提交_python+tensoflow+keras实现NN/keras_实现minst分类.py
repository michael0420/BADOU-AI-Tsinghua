import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 加载数据集
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels.shape  = ', train_labels.shape)
print('test_images.shape = ', test_images.shape)
print('test_labels.shape = ', test_labels.shape)

# 打印第一张图片
import matplotlib.pyplot as plt

photo = test_images[0]
plt.imshow(photo,cmap='binary')
plt.show()

from tensorflow.keras import models
from tensorflow.keras import layers
# 搭建网络结构
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

# 网络编译
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# 数据预处理  对数据进行归一化处理
train_images = train_images.reshape(60000,28*28)
train_images = train_images.astype('float32') /255

test_images = test_images.reshape(10000,28*28)
test_images = test_images.astype('float32') / 255

# 对标签进行one_hot编码
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# 网络训练
history = network.fit(train_images,train_labels,epochs=5,batch_size=128)
y = network.summary()

# 显示训练loss和acc曲线
# 显示训练集和验证集的acc和loss曲线
acc = history.history['acc']

loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# 模型测试评估
#输出损失和精确度
score = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss:',score[0])
print('test_acc',score[1])
# verbose：控制日志显示的方式
# verbose = 0  不在标准输出流输出日志信息
# verbose = 1  输出进度条记录

# 预测图片
import numpy as np
photo = test_images[100]

photo = photo.reshape((1,28*28))  # 更改为与输入数据一样的张量格式

photo_label = test_labels[100]

photo_label = np.argmax(photo_label)
prident_label = network.predict(photo)
prident_label = np.argmax(prident_label)

print("真实图片标签为：",photo_label)
print(('预测结果标签为：',prident_label))