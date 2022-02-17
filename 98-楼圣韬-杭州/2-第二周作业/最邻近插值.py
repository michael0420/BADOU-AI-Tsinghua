import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imutils
img=cv2.imread("lenna.png")

def nearinterp(img,dh,dw):
    h,w,c=img.shape
    dstimg=np.zeros((dh,dw,c),np.uint8)
    sh=dh/h
    sw=dw/w
    for i in range(dh):
        for j in range(dw):
            x=int(i/sh)
            y=int(j/sw)
            dstimg[i,j]=img[x,y]
    return dstimg
dst=nearinterp(img,900,900)

cv2.imshow("origin",img)
cv2.imshow("nearest interp",dst)
cv2.waitKey(0)


mpl.rcParams['font.sans-serif'] = 'KaiTi'
fig = plt.figure()
plt.suptitle("最邻近插值")
plt.subplot(211)
plt.title("原图")
plt.imshow(imutils.opencv2matplotlib(img))

plt.subplot(212)
plt.title("插值图")
plt.imshow(imutils.opencv2matplotlib(dst))
plt.show()