# import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
import matplotlib.pyplot as plt
# import cv2
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import to_categorical


# 1. 导入数据并查看

(trainImages,trainLables),(testImages,testLables)=mnist.load_data()
# print('trainImages.shape:',trainImages.shape)
# print('trainLables:',trainLables)
# print('testImages.shape:',testImages.shape)
# print('testLables:',testLables)
#查看测试集第一张图片
# test0=testImages[0]
# plt.imshow(test0,cmap=plt.cm.binary)
# plt.show()
#cv2.imshow('test0',test0)
#cv显示较小




#2. 使用keras构建神经网络

net=models.Sequential()
net.add(layers.Dense(units=512,activation='relu',input_shape=(28*28,)))
net.add(layers.Dense(units=10,activation='softmax'))
net.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#数据重塑与归一化
trainImages=trainImages.reshape((60000,28*28))
trainImages=trainImages.astype('float32')/255
testImages=testImages.reshape((10000,28*28))
testImages=testImages.astype('float32')/255
#标签独热化
testLablesOriginal=testLables
trainLables=to_categorical(trainLables)
testLables=to_categorical(testLables)

#3. 训练与测试

net.fit(trainImages,trainLables,epochs=7,batch_size=128)

testLoss,testAccuracy=net.evaluate(testImages,testLables,verbose=1)
print('\nTest Loss:',testLoss,'Test Accuracy:',testAccuracy)

result=net.predict(testImages)


