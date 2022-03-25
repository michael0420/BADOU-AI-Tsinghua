import random

import cv2


def GaussianNoise(img, means, sigma, percetage):
    noise_num = img.shape[0] * img.shape[1] * percetage
    for i in range(int(noise_num)):
        h = random.randint(0, img.shape[0] - 1)
        w = random.randint(0, img.shape[1] - 1)
        result = img[h, w] + random.gauss(means, sigma)
        if result < 0:
            result = 0
        if result > 255:
            result = 255
        img[h, w] = result
    return img


def SaltNoise(img, percetage):
    noise_num = img.shape[0] * img.shape[1] * percetage
    for i in range(int(noise_num)):
        h = random.randint(0, img.shape[0] - 1)
        w = random.randint(0, img.shape[1] - 1)
        random_num = random.random()
        if random_num > 0.5:
            img[h, w] = 0
        else:
            img[h, w] = 255
    return img


# cv2.imread 重复执行是因为两个加噪音的函数会改变原图
img_src = cv2.imread("lenna.png", 0)
img1 = GaussianNoise(img_src, 20, 4, 0.8)  # 这里用20,肉眼能明显看到效果,老师的代码用2太小了
img_src = cv2.imread("lenna.png", 0)
img2 = SaltNoise(img_src, 0.2)
img_src = cv2.imread("lenna.png", 0)

# show
cv2.imshow("src", img_src)
cv2.imshow("Gaussian", img1)
cv2.imshow("Salt", img2)
cv2.waitKey(0)

'''
使用api:
from skimage import util
img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='poisson')  泊松噪音
'''
