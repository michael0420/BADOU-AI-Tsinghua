# 使用sklearn 实现 噪声调用
import  cv2
import  numpy as np
from skimage import  util

img = cv2.imread("lenna.png")
# noise_gs_img = util.random_noise(img,mode="salt")
noise_gs_img = util.random_noise(img,mode="poisson")

cv2.imshow("img",img)
cv2.imshow("noise",noise_gs_img)

cv2.waitKey(0)