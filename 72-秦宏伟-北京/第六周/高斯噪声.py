import cv2
import random
"""
img,待加噪声的灰度图
means,高斯噪声中值
sigma,
SNR，信噪比
"""
def gauss(img,means,sigma,SNR):
    size_x = img.shape[0]
    size_y = img.shape[1]
    num_gs = int(size_x*size_y*SNR)
    for i in range(num_gs):
        rand_x = int(size_x * random.random())
        rand_y = int(size_y * random.random())
        rand = int(random.gauss(means,sigma))
        print(rand)
        new_gray = img[rand_x,rand_y]+int(random.gauss(means,sigma))
        # print('old img %s and new img %s'%(str(img[rand_x,rand_y]),str(new_gray)))
        if new_gray > 255 :
            new_gray = 255
        elif new_gray <0:
            new_gray = 0
        img[rand_x,rand_y] = int(new_gray)
    return img


img = cv2.imread('lenna.png',0)

cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('input_image', img)

means = 0
sigma = 5
SNR = 0.2
img_new = gauss(img,means,sigma,SNR)

cv2.namedWindow('gauss_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('gauss_image', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
