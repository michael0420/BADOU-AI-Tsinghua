import numpy as  np
import  cv2
from numpy import shape
import  random

img =cv2.imread("lenna.png",0) # 512* 512
# 定义 sigma and  means 初始值
means =0
sigma =5
rows,cols = img.shape
# 创建一张新的图像
new_img = np.zeros((rows,cols),img.dtype)

for i in range(rows):
    for j in range(cols):
        # Pout = Pin + random.gauss
        # random.gauss(means,sigma) 生成高斯随机数
        new_img[i,j] = img[i,j]+random.gauss(means,sigma)
        # 重新缩放像素值 0~255
        if new_img[i,j] <0:
            new_img[i, j] =0
        elif new_img[i,j] >255:
            new_img[i, j]=255


cv2.imshow("img",img)
cv2.imshow("new_img",new_img)
cv2.waitKey(0)