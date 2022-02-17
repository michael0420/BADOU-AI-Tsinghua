import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
(train_images,train_labels),(test_images,test_labels)=mnist.load_data() # 读取数据
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)
'''
digit=test_images[0]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
'''
network = models.Sequential()  # 串联操作，顺序执行网络结构
# layers:表示神经网络中的数据处理层 Dense 表示该层为全连接层 activation为选择激活函数
# input_shape(28*28,): 表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，后面的“,”表示一个元素到底包含多少数字都没有关系
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
# optimizer 为优化器  loss为损失函数的选择   metrics 为评估性能指标
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# reshape 把(28,28)的二维数组变为28*28的一维度数组
# astype 把0—255的像素点转换成0-1之间的浮点值
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

# 转化labels 转化成one-hot 特征，即7->[0,0,0,0,0,0,0,1,0,0]

print("before changes:",test_labels[0])
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
print("after changes:",test_labels[0])

network.fit(train_images,train_labels,epochs=5,batch_size=128)  # 开始训练

print('done')
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
print('test_acc:%.2f'%(test_acc*100),"%")

# 测试
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
digit = test_images[1]
plt.imshow(digit,cmap=plt.cm.binary)
test_images=test_images.reshape((10000,28*28))
res = network.predict(test_images)
for i in range(res[1].shape[0]):
    if res[1][i] ==1:
        plt.title("the number for the picture is : %d"%i)
        break
plt.show()