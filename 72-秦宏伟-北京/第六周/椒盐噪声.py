import cv2
import random
"""
给图像加椒盐噪声
SNR信噪比
"""
def jy(img,SNR):
    noise_img = img
    x_size = img.shape[0]
    y_size = img.shape[1]
    noise_num = int(x_size*y_size*SNR)

    for i in range(noise_num):
        rand_x= int(x_size*random.random())
        rand_y = int(y_size*random.random())
        rand_choice = random.random()
        if rand_choice > 0.5:
            noise_img[rand_x,rand_y] = 0
        else:
            noise_img[rand_x, rand_y] = 255
    return noise_img

img = cv2.imread('lenna.png',0)
cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('input_image', img)

SNR = 0.2
img_new = jy(img,SNR)

cv2.namedWindow('noise_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('noise_image', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()