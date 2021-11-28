import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img=cv2.imread("lenna.png")
# print(img.shape)
# image->gray
length=img.shape[0]
width=img.shape[1]
img_gray = np.zeros([length,width],img.dtype)
for i in range(length):
    for j in range(width):
        #浮点算法
        img_gray[i][j]=img[i][j][0]*0.11+img[i][j][1]*0.59+img[i][j][2]*0.3
        #整数算法
        #img_gray[i][j]=(img[i][j][0]*11+img[i][j][1]*59+img[i][j][2]*30)/100
        #移位算法
        # img_gray[i][j]=(img[i][j][0]*28+img[i][j][1]*151+img[i][j][2]*76)>>8
        #平均值法
        # img_gray[i][j]=(img[i][j][0]+img[i][j][1]+img[i][j][2])/3
        #仅取绿色
        # img_gray[i][j]=img[i][j][2]
print (img_gray)
plt.imshow(img_gray, cmap='gray')
plt.show()
# (2)使用rgb2gray或cv2.cvtColor方法
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.show()


