

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread('lenna.png')
h,w = img.shape[:2]

'''
[0:2]是切片的意思，.shape 应当是OpenCV模块中处理图片的，是图片的一个属性，这个属性是个列表 ，然后对这个列表切片操作。
例子：h,w = img.shape[:2] 获取彩色图片的高、宽，并且赋值给h和w；如果是h,w,v = img.shape[:3] 获取彩色图片的高、宽、通道，并赋值给h w v
'''

'''
另一种获取方法：
img = Image.open('lenna.png')
imgSize = img.size  #大小/尺寸
w = img.width       #图片的宽
h = img.height      #图片的高
f = img.format      #图像格式
'''
#两种方法都可以获取图片宽高，但获取和读入必须对应，不能用CV读入，用PIL取值。

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        #不能用PIL读入，不是numpy格式

img_binary = np.where(img_gray/255 >= 0.5, 1, 0)

'''
cv2.imshow("image show gray",img_gray)
cv2.waitKey(0)
'''

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
plt.imshow(img_gray, cmap = 'gray_r') #灰度反转

plt.subplot(224)
plt.imshow(img_binary, cmap='gray')

plt.show()      #显示图像必须加这句





