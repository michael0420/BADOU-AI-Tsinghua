"""
@author: 14+陈志海+广东
fcn: gauss_noise: 图像添加高斯噪声
fcn：pepper_noise: 图像添加白噪声
"""

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def gray_cvt(data):
    result = data
    for i in range(result.size):
        if result[i] < 0:
            result[i] = 0
        elif result[i] > 255:
            result[i] = 255
    return result


def gauss_noise(src, percentage, mean, sigma):
    if percentage < 0 or percentage > 100:
        print("percentage is out of range[0, 100]")
        return -1
    result = src.copy()
    # result = cv2.copyTo(src, None)
    num_noise = int(0.01 * percentage * src.shape[0] * src.shape[1])
    for i in range(num_noise):
        hi = random.randint(0, src.shape[0]-1)
        wi = random.randint(0, src.shape[1]-1)
        result[hi, wi] = result[hi, wi] + np.random.normal(mean, sigma, 3)
        result[hi, wi] = gray_cvt(result[hi, wi])

    return result


def pepper_noise(src, percentage):
    result = src.copy()
    num_noise = int(0.01 * percentage * src.shape[0] * src.shape[1])
    pepper = [[0, 0, 0], [255, 255, 255]]
    for i in range(num_noise):
        hi = random.randint(0, src.shape[0]-1)
        wi = random.randint(0, src.shape[1]-1)
        result[hi, wi] = pepper[random.randint(0, 1)]

    return result


img_bgr = cv2.imdecode(np.fromfile("lenna.png", dtype=np.uint8), -1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gauss = gauss_noise(img_rgb, percentage=30, mean=10, sigma=30)
img_pepper = pepper_noise(img_rgb, 10)

plt.figure()
plt.subplot(131)
plt.imshow(img_rgb)
plt.axis("off")
plt.title("lenna")

plt.subplot(132)
plt.imshow(img_gauss)
plt.axis("off")
plt.title("lenna_gauss_noise")

plt.subplot(133)
plt.imshow(img_pepper)
plt.axis("off")
plt.title("lenna_pepper_noise")

plt.show()
