import numpy as np
import cv2
from numpy import shape
import random
from skimage import util

def  salt_pepper(src,percetage):
    sumnum=int(percetage*src.shape[0]*src.shape[1])
    print(sumnum)
    for i in range(sumnum):
	    randX=random.randint(0,src.shape[0]-1)
	    randY=random.randint(0,src.shape[1]-1)
	    if random.random()<=0.5:
	    	src[randX,randY]=0
	    else:
	    	src[randX,randY]=255
    return src

img=cv2.imread('lenna.png',0)
img1=salt_pepper(img,0.1)
#在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
#cv2.imwrite('lenna_PepperandSalt.png',img1)
cv2.imshow('source',img1)
cv2.waitKey(0)


img2=cv2.imread('lenna.png',0)
img3=util.random_noise(img2,mode='s&p',amount=0.1)
cv2.imshow('source',img3)
cv2.waitKey(0)
'''
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
功能：为浮点型图片添加各种随机噪声
参数：
image：输入图片（将会被转换成浮点型），ndarray型
mode： 可选择，str型，表示要添加的噪声类型
	gaussian：高斯噪声
	localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
	poisson：泊松噪声
	salt：盐噪声，随机将像素值变成1
	pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
	s&p：椒盐噪声
	speckle：均匀噪声（均值mean方差variance），out=image+n*image
seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果谁False，则输出矩阵的值可能会超出[-1,1]
mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
--------
返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
'''