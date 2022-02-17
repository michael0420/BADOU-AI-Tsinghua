

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("lenna.png")

h,w= img.shape[:2]
image_gray=np.zeros([h,w],img.dtype)

for i in range(h):
    for j in range(w):
        m=img[i,j]
        #print(m)
        # 浮点运算 rgb转gray
        image_gray[i,j]=int(m[0]* 0.11 + m[1] * 0.59 + m[2] * 0.3)
        # 整数方法
        image_gray[i, j] = int((m[0] * 11 + m[1] * 59 + m[2] * 30)/100)

        # 平均方法 RuntimeWarning: overflow encountered in ubyte_scalars
        #image_gray[i, j] = int((m[0] + m[1] + m[2]) / 3)
        #print(image_gray[i,j])

#print("image show gray: %s"%image_gray) np.array(m[0]*0.11,m[1]*0.59,m[2]*0.3)
#cv2.imshow("gray image",image_gray)
#cv2.waitKey(0) # 等待输入， 避免自动关闭


plt.subplot(221)
img=plt.imread("lenna.png")
plt.imshow(img)
plt.show()
#print(img)

# 灰度话
# img

#image_gray= rgb2gray(img)
image_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray)
plt.show()


# 二级化
img_01= np.where(image_gray>=0.5,1,0)
plt.imshow(img_01)
plt.show()


