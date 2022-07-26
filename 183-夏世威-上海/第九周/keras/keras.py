from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 网络模型
model = models.Sequential()
# 卷积 kernel_size(3,3) 输出特征数32
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# Relu 激活层
model.add(layers.Activation('relu'))
# 构建池化层
model.add(layers.MaxPool2D(pool_size=(10, 10), strides=(1, 1)))

model.add(layers.Conv2D(64, (5, 5)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1)))

# 全连接
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 输出模型的结构信息
model.summary()
plot_model(model, to_file='mnist.png', show_shapes=True)
# 优化器选RMSProp 比SGD好在可以防止抖动
# loss 选 交叉熵
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1, batch_size=64)
result = model.evaluate(test_images, test_labels)
print(result)
