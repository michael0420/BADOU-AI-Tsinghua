#!/usr/bin/env python 
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, Input, Model
from keras.datasets import mnist
from keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Reshape
from keras.optimizers import Adam


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        # 判别网络
        self.discriminator = self.build_discriminator()
        # 编译 设置损失函数 优化器等
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # 生成网络
        self.generator = self.build_generator()

        # 输入
        z = Input(shape=(self.latent_dim,))
        # 生成图像
        img = self.generator(z)
        # 将生成的图像输入判别网络
        validity = self.discriminator(img)
        # 串联的模型 不需要训练判别网络
        self.discriminator.trainable = False
        # 串联模型
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        return model

    def train(self, epochs, batch_size, sample_interval):
        # 加载数据
        (X_train, _), (_, _) = mnist.load_data()
        # 归一化到[-1, 1]
        X_train = X_train / 127.5 - 1
        # 扩充维度
        X_train = np.expand_dims(X_train, axis=3)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # 随机选择一批图像
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # 随机向量生成一批图像
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # 训练判别网络
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            # 正负样本的损失各占一半
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成网络
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_loss = self.combined.train_on_batch(noise, valid)

            print("%d[D loss:%f,acc:%.2f%%][G loss:%f]" % (epoch, d_loss[0], d_loss[1] * 100, gen_loss))
            # 保存图像
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # 缩放至[0,1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                # 关闭坐标刻度值
                axs[i, j].axis('off')
                cnt += 1
        # 保存图像
        fig.savefig('./images/mnist_%d.png' % epoch)
        # 防止循环时图像覆盖
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)
