import numpy as np
import cv2


# 均值哈希算法
def aHash(src_img):
    tmp_img = cv2.resize(src_img, (8, 8), interpolation=cv2.INTER_CUBIC)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(tmp_img)
    ahash = ''
    for i in range(8):
        for j in range(8):
            if tmp_img[i, j] > mean:
                ahash = ahash + '1'
            else:
                ahash = ahash + '0'
    return ahash


# 差值哈希算法
def dHash(src_img):
    tmp_img = cv2.resize(src_img, (9, 8), interpolation=cv2.INTER_CUBIC)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    dhash = ''
    for i in range(8):
        for j in range(8):
            if tmp_img[i, j] > tmp_img[i, j + 1]:
                dhash = dhash + '1'
            else:
                dhash = dhash + '0'
    return dhash


# 感知哈希算法
def pHash(src_img):
    tmp_img = cv2.resize(src_img, (32, 32), interpolation=cv2.INTER_CUBIC)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(tmp_img))
    dct_roi = dct[:8, :8]
    mean = np.mean(dct_roi)
    phash = ''
    for i in range(8):
        for j in range(8):
            if dct_roi[i, j] > mean:
                phash = phash + '1'
            else:
                phash = phash + '0'
    return phash


# 计算两个hash值的汉明距离和相似度
def calc_hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        raise ValueError("hash1={}, hash2={}, len(hash1) != len(hash2)")
    hamming_dist = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            hamming_dist = hamming_dist + 1
    # 相似度
    similarity = 1 - hamming_dist * 1.0 / 64
    return hamming_dist, similarity
