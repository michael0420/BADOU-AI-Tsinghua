"""
@author: 14+陈志海+广东
@fcn：aHash，dHash，pHash
"""
import cv2
import numpy as np


def aHash(img):
    img_resize = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    avg = np.mean(np.mean(img_gray, axis=0), axis=0)
    img_gray[img_gray <= avg] = 0
    img_gray[img_gray > avg] = 1
    str = ''
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            str += "%d" % img_gray[i, j]

    return str


def dHash(img):
    img_resize = cv2.resize(img, dsize=(9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_d = img_gray[:, :8] - img_gray[:, 1:]
    img_d[img_d > 0] = 1
    img_d[img_d < 0] = 0
    str = ''
    for i in range(img_d.shape[0]):
        for j in range(img_d.shape[1]):
            str += "%d" % img_d[i, j]

    return str


def pHash(img):
    img_resize = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_float = np.zeros(shape=img_gray.shape, dtype=np.float32)
    img_dct = cv2.dct(img_float)
    img_ahash = img_dct[:8, :8]
    avg = np.mean(np.mean(img_ahash, axis=0), axis=0)
    img_ahash[img_ahash <= avg] = 0
    img_ahash[img_ahash > avg] = 1
    str = ''
    for i in range(img_ahash.shape[0]):
        for j in range(img_ahash.shape[1]):
            str += "%d" % img_ahash[i, j]

    return str


def hamming_distance(hash1, hash2):
    """
    sum the hamming distance of two hash_str
    :param hash1: hash string 1
    :param hash2: hash string 2
    :return: 
    """
    if len(hash1) != len(hash2):
        print("string length must be same.")
        return -1
    distance = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            distance += 1
    similarity = 1 - distance / len(hash1)

    return similarity, distance


# main()
lenna = cv2.imdecode(np.fromfile("lenna.png", dtype=np.uint8), -1)
lenna_noise = cv2.imdecode(np.fromfile("lenna_noise.png", dtype=np.uint8), -1)

# aHash
str1 = aHash(lenna)
str2 = aHash(lenna_noise)
similarity, distance = hamming_distance(str1, str2)
print("aHash method: similarity is %.2f, hamming distance is %d" % (similarity, distance))

# dHash
str1 = dHash(lenna)
str2 = dHash(lenna_noise)
similarity, distance = hamming_distance(str1, str2)
print("dHash method: similarity is %.2f, hamming distance is %d" % (similarity, distance))

# pHash
str1 = pHash(lenna)
str2 = pHash(lenna_noise)
similarity, distance = hamming_distance(str1, str2)
print("pHash method: similarity is %.2f, hamming distance is %d" % (similarity, distance))
