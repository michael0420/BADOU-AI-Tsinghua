import cv2
import numpy as np

####################################################
# sobel  算子
img = cv2.imread("lenna.png",0)
#print(img.shape)
"""
sobel 函数求导后会有负值出来，以及大于255 的值
原因是图像是uint8 ，8维无 符号数（范围在[0,255]）
所以sobel 建立的图像位数不够，会有截断
用16位有符号的数据类型，即cv2.CV_16s
"""

# 参数1 ： 输入原图
# 参数2 ： 图像的深度  -1 表示与原图像相同的深度
# 参数3 ： dx=1,dy=0 表示在x 方向求一阶导数
#         dx=0,dy=1 表示在y 方向求一阶导数
# 如果同时 求一阶导数，通常得不到想要的结果
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
#print(x)

"""
经过处理后的图像，要把他还原到uint8 形式，用convertScaleAbs()
否则将无法显示图像，而只是一幅灰色窗口

"""
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
"""
由于sobel 算子是在两个方向计算的，最后还需要用cv2.addWeighted() 函数将其组合起来
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) 
alpha ,beta 都是权重，alpha 是第一个权重，beta  是第二个权重
gamma 是加到最后结果上的一个值
"""
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

cv2.imshow("dst",dst)
cv2.imshow("absx",absX)
cv2.imshow("absy",absY)

cv2.waitKey(0)