#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 21:27
# @Author  : ystbr
# @FileName: myGAN.py

# 让python2的使用跟python3相同
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 他喵的居然是GPU版本代码，GAN一定要用GPU吗，这么耗费内存的嘛
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class myGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # 启用Adam优化器，参数1->学习率，参数2->beta1->一阶矩估计的知数衰减率->梯度的均值，负责梯度的方向
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        self.discriminator.trainable = False

        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        vivid = self.discriminator(img)

        # --------------------------------------------------------------------------------------------------------------
        # 训练生成器->试图“以假乱真”，骗过鉴定器
        self.combined = Model(z,vivid)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)

    # ------------------------------------------------------------------------------------------------------------------
    # 构建“生成器”
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
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    # ------------------------------------------------------------------------------------------------------------------
    # 构建“鉴别器”
    def build_discriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # ------------------------------------------------------------------------------------------------------------------
    # 开始训练，监督两个loss
    def Train(self,epochs,batch_size=128,sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        real = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):
            index = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[index]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            # 对两个loss取平均值
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            g_loss = self.combined.train_on_batch(noise, real)

            # 可视化两个loss
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # 在一定的间隔时间后保存“生成器”生成的图像
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # ------------------------------------------------------------------------------------------------------------------
    #
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # 归一化
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = myGAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)




