
import cv2
import numpy as np
def function (img):
    height,width,channels =img.shape
    emptyimage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh)
            y=int(j/sw)
            emptyimage[i,j]=img[x,y]
    return emptyimage

img=cv2.imread("/Users/oh/Desktop/zuoye/WechatIMG13.jpeg")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("zuilinjinchazhi.jpeq",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)






