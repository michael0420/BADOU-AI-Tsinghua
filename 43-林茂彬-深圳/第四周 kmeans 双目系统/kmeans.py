# coding: utf-8
# https://blog.csdn.net/lanshi00/article/details/104109963

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread('lenna.png', 0)
'''
imread函数有两个参数，第一个参数是图片路径，第二个参数表示读取图片的形式，有三种：
cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
cv2.IMREAD_UNCHANGED：包括alpha，可以直接写-1
'''
'''
retval, bestLabels, centers = cv2.kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
函数参数：
data:  需要分类数据，最好是np.float32的数据，每个特征放一列。
K:  聚类个数 
bestLabels：预设的分类标签或者None
criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon） 其中，type有如下模式：
cv2.TERM_CRITERIA_EPS ：精确度（误差）满足epsilon，则停止。
cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter，则停止。
cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER：两者结合，满足任意一个结束。
attempts：重复试验kmeans算法次数，将会返回最好的一次结果
flags：初始中心选择，可选以下两种：
v2.KMEANS_PP_CENTERS：使用kmeans++算法的中心初始化算法，即初始中心的选择使眼色相差最大.详细可查阅kmeans++算法。(Use kmeans++ center initialization by Arthur and Vassilvitskii)
cv2.KMEANS_RANDOM_CENTERS：每次随机选择初始中心（Select random initial centers in each attempt.）
'''
'''
返回值：
compactness：紧密度，返回每个点到相应重心的距离的平方和
labels：结果标记，每个成员被标记为分组的序号，如 0,1,2,3,4...等
centers：由聚类的中心组成的数组
'''

#data:  需要分类数据，最好是np.float32的数据，每个特征放一列。

rows, cols = img.shape[:] #获取图像高度、宽度
data = img.reshape((rows * cols, 1)) #图像二维像素转换为一维
data = np.float32(data)

#K:  聚类个数
k=4

#criteria：迭代停止的模式选择
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
attempts = 10


#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS


#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, flags)



#生成最终图像

dst = labels.reshape((img.shape))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = ['原始图像', '聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'), #使用plt.subplot来创建小图. plt.subplot(221)表示将整个图像窗口分为2行2列, 当前位置为1.第一个参数代表子图的行数；第二个参数代表该行图像的列数； 第三个参数代表每行的第几个图像。
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()