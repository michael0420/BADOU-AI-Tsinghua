'''
最邻近差值法
当图像增大时，向图像中插入像素点遵循此法。
插入像素点的位置由图像增大的倍数决定
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt


def interp(img,w,h):
    height,width,channels=img.shape
    emptyImage=np.zeros([h,w,channels],np.uint8)
    x_h=h/height
    x_w=w/width
    for i in range(h):
        for j in range(w):
            _h=int(i/x_h)
            _w=int(j/x_w)
            emptyImage[i,j]=img[_h,_w]
    return emptyImage

img=cv2.imread('lenna.png')
# cv2.imshow("1",img)
newImg=interp(img,2048,2048)
# cv2.imshow("2",newImg)

plt.subplot(121)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(newImg,cv2.COLOR_BGR2RGB))
plt.show()
cv2.waitKey(0)

