# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:28:29 2021

@author: Administrator
"""
import cv2
import numpy as np


class myHash():
    def __init__(self, path):
        self.img = cv2.imread(path)

    def meanHash(self):
        img = cv2.resize(self.img, (8, 8), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        value = 0
        key = ''
        for i in range(8):
            for j in range(8):
                value += gray[i][j]
        avg = value / 64
        for i in range(8):
            for j in range(8):
                if gray[i][j] > avg:
                    key = key + '1'
                else:
                    key = key + '0'
        return key

    def diffHash(self):
        img = cv2.resize(self.img, (9, 8), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key = ''
        for i in range(8):
            for j in range(8):
                if gray[i][j] > gray[i][j + 1]:
                    key += '1'
                else:
                    key += '0'
        return key

    def comHash(self, key1, key2):
        if len(key1) != len(key2):
            return -1
        n = 0
        for i in range(len(key1)):
            if key1[i] != key2[i]:
                n += 1
        return n


def main():
    path1 = '/home/uers/desk_B/八斗/lenna.png'
    path2 = '/home/uers/desk_B/八斗/lenna_noise.png'
    myhash1 = myHash(path1)
    myhash2 = myHash(path2)
    key1 = myhash1.meanHash()
    key2 = myhash2.meanHash()
    n = myhash1.comHash(key1, key2)
    print('key1 is: ', key1)
    print('key2 is: ', key2)
    print('均值哈希算法相似度：', n)

    key1 = myhash1.diffHash()
    key2 = myhash2.diffHash()
    n = myhash1.comHash(key1, key2)
    print('key1 is: ', key1)
    print('key2 is: ', key2)
    print('差值哈希算法相似度：', n)


if __name__ == "__main__":
    main()
