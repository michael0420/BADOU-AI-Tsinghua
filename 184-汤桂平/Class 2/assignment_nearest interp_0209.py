import cv2
import numpy as np

def funct(img):
    height,width,channels=img.shape
    Eh=800/height
    Ew=800/width
    EmptyImg=np.zeros((800,800,channels),np.uint8)  # 开始写成了np.zeros((800,800,channels),img.dtype)
    for i in range(800):
        for j in range(800):
            x=int(i/Eh)
            y=int(j/Ew)
            EmptyImg[i,j]=img[x,y]
    return EmptyImg

img=cv2.imread("lenna.png")
Zoom=funct(img)
print("Zoomed img shows:",Zoom)
print("src img shows:",img)
cv2.imshow("Zoomed img",Zoom)  # imshow函数第一个参数为输出图片的title
cv2.imshow("source img",img)
cv2.waitKey(0)  # 没有这一行代码的话图片将显示一下后即可关闭