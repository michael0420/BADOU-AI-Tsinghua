
'''
1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
3.随机获取要加噪的每个像素位置P（i, j）
4.指定像素值为255或者0。
5.重复3, 4两个步骤完成所有NP个像素的加噪

'''

import numpy as np
import cv2
from numpy import shape
import random

#制造椒盐噪声
def pepper_salt_noise(src_img,percentage):

    NoiseImg = src_img
	## 2.计算总像素
	#Sum_pixel = src_img.shape[0] * src_img.shape[1]

	#3.1 总像素 * 信噪比 = 添加噪声的像素数量
    NoiseNum = int(percentage * src_img.shape[0] * src_img.shape[1])

	#4. 添加噪声
    for i in range(NoiseNum):
	#每次取一个随机点
    #按行列表示方式，取像素点
    #random.randint()生成随机整数
    #椒盐噪声图片边缘不处理，故-1
	    x = random.randint(0,src_img.shape[0]-1)
	    y = random.randint(0,src_img.shape[1]-1)
	    #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
		#自己设定条件，做为添加噪声的判断依据
	    if random.random()<=0.5:
	    	NoiseImg[x,y]=0
	    else:
	    	NoiseImg[x,y]=255


    return NoiseImg

img=cv2.imread('F:/cycle_gril/lenna.png',0)
#1.指定信噪比，信号和噪声的比例[0,1]
percentage = 0.1
img1 = pepper_salt_noise(img,percentage)

img = cv2.imread('F:/cycle_gril/lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)