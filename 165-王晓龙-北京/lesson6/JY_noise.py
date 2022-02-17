#
import  numpy as np
import  cv2
import math
import  random

src = cv2.imread("lenna.png",0)
cv2.imshow("src",src)
img =src
rows,cols = img.shape

SNR =0.8 # SNR 信噪比
SP = rows*cols # 像素点数目
NP = int(SP*SNR) # 要加噪的像素数目 取整数
# 循环 加噪的个数
for i in range(NP):
    # 随机取x,y
    randX = np.random.randint(0, rows)  # 0~512
    randY = np.random.randint(0, cols)  # 0~512

    if random.random() <=0.5:
        img[randX,randY] =0
    else:
        img[randX,randY] =255

cv2.imshow("img",img)

cv2.waitKey(0)