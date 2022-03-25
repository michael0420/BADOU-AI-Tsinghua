import os
import random
import cv2
import numpy as np


def transform_img(img):
    img = cv2.resize(img, (224, 224))  # 固定大小 224*224
    img = np.transpose(img, (2, 0, 1))  # H*W*C --> C*H*W
    img.astype("float32")
    img = img / 255.0  # [0.0,255.0] --> [0.0, 1.0]
    return img


# 训练集的数据加载器,label在文件名里面
def data_loader(data_dir, batch_size=10, model="train"):
    file_names = os.listdir(data_dir)

    if model == "train":
        random.shuffle(file_names)
    batch_imgs = []
    batch_labels = []
    for name in file_names:
        filepath = os.path.join(data_dir, name)
        img = cv2.imread(filepath)
        img = transform_img(img)
        if name[0] == "H" or name[0] == "N":
            label = 0  # 高度近视和正常的都是负样本
        elif name[0] == "P":
            label = 1
        else:
            raise "无效的label"
        batch_imgs.append(img)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            imgs_array = np.array(batch_imgs).astype("float32")
            labels_array = np.array(batch_labels).astype("int64")
            yield imgs_array, labels_array
            batch_imgs = []
            batch_labels = []
    # 最后一批,不够batch_size的情况
    if len(batch_imgs) > 0:
        imgs_array = np.array(batch_imgs).astype("float32")
        labels_array = np.array(batch_labels).astype("int64")
        yield imgs_array, labels_array


# 验证集的数据加载器,label在labels.csv里面
def valid_data_loader(data_dir, csvfile, batch_size=10):
    filelists = open(csvfile).readlines()
    filelists = filelists[1:]  # 第一行表头不要

    batch_imgs = []
    batch_labels = []
    for line in filelists:
        line = line.strip().split(",")
        img_name = line[1]
        label = int(line[2])
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path)
        img = transform_img(img)
        batch_imgs.append(img)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            imgs_array = np.array(batch_imgs).astype("float32")
            labels_array = np.array(batch_labels).astype("float32").reshape(-1, 1)
            yield imgs_array, labels_array
            batch_imgs = []
            batch_labels = []
    # 最后一批,不够batch_size的情况
    if len(batch_imgs) > 0:
        imgs_array = np.array(batch_imgs).astype("float32")
        labels_array = np.array(batch_labels).astype("float32").reshape(-1, 1)
        yield imgs_array, labels_array
