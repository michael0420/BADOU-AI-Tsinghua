import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#读取原始图像
img = cv.imread('../../../../../BaiduNetdiskDownload/lenna.png')
print(img.shape)


#图像二维像素转化为一维
data = img.reshape((-1, 3))
data = np.float32(data)

#停止条件 (type, max_iter, epsilon)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv.KMEANS_RANDOM_CENTERS

#2类
compactness, labels2, centers2 = cv.kmeans(data, 2, None, criteria, 10, flags)
#4类
compactness, labels4, centers4 = cv.kmeans(data, 4, None, criteria, 10, flags)
#8类
compactness, labels8, centers8 = cv.kmeans(data, 8, None, criteria, 10, flags)
#16类
compactness, labels16, centers16 = cv.kmeans(data, 16, None, criteria, 10, flags)

#图像转换回uint8二维类型

centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
print("111", centers4)
print("labels", labels4.flatten())
res = centers4[labels4.flatten()]
print()
print(res)
dst4 = res.reshape((img.shape))

#图像转换为RGB显示
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
dst2 = cv.cvtColor(dst2, cv.COLOR_BGR2RGB)
dst4 = cv.cvtColor(dst4, cv.COLOR_BGR2RGB)


#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']
images = [img, dst2, dst4]
for i in range(3):
    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
