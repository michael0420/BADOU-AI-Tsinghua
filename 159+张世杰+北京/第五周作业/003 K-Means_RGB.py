# encoding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 1)
'''把每个通道都分别转换为一维，得到 （512*512 ， 3）矩阵， 同时转换为浮点型数值，方便计算 '''
data = img.reshape(-1, 3)
data = np.float32(data)

'''为cv2.KMeans设置终止条件：迭代次数为10或数据误差为1个像素点；设置标签：随机选择初始质心'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

'''设置聚类：2、4、8、16、64'''
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

'''将图像转回uint8：首先将质心转换为uint8，然后将同一类的像素点赋值为质心像素值, 最后将图像形状还原'''
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]  # 这种语法不明白
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))
'''直接用cv显示'''
# cv2.imshow('2', dst2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''用PLT显示'''
'''转换图像格式：BGR to RGB'''
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

'''显示图像：pycharm titles 不能显示汉字'''
titles = [u'YS', u'JL K=2', u'JL K=4',
          u'JL K=8', u'JL K=16', u'JL K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
