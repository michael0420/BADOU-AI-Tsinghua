import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1、读取图像
img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE) # 读取灰度图像
print(img.shape)

img_rows = img.shape[0] # 图像行数
img_clos = img.shape[1] # 图像列数

'''
cv2.kmeans函数原型如下:
kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
    返回值：
    compactness：紧密度，返回每个点到对应中心的距离平方和
    labels：结果标记, 每个成员标记为0,1等
    centers:聚类中心组成的数组

    参数：
    data: 需要进行聚类的N维数据，float32类型的数据。
    K：需要分裂出的簇数，也就是K值
    bestLabels：预设的分类标签，没有就是None
    criteria：聚类终止条件，这是一个含有三个元素的元组数据类型，格式为(type,max_iter,epsilon)，type有：
             cv2.TERM_CRITERIA_EPS   精确度(误差)满足epsilon则停止，也就是每个点离质心的距离，也就是误差平方和(SEE)，
             cv2.TERM_CRITERIA_MAX_ITER 迭代次数超过max_iter则停止
             cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER 两者结合，只要满足一个就结束
    attempts：k-means算法重复次数，返回数据聚类最好的那一次结果
    flags：初始类中心选择方式，有三种方法：
            c2.KMEANS_RANDOM_CENTERS   随机选择中心
            cv2.KMEANS_PP_CENTERS       使用kmeans++初始化中心
            cv2.KMEANS_USE_INITIAL_LABELS
            
    centers:簇中心的输出矩阵，每个簇有一行数据
    
'''
# 图像数据转化为1维
data = img.reshape((img_rows * img_clos, 1))
data = np.float32(data) # 转换为32位浮点数据，kmeans的数据最好用32位浮点

# 停止条件(type, max_inter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 初始类中心
flags = cv2.KMEANS_RANDOM_CENTERS

# 进行K-means聚类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
print(centers)

# 生成最终的图像
dst = labels.reshape((img.shape[0], img.shape[1])) # 将labels转换为行列的图像数据
print(dst)

# 显示图片
plt.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示中文
plt.figure(figsize=(6, 6), dpi=100)  # 画布10*10寸，dpi=100
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 子图横竖间隔0.3英寸

# 显示原始灰度图
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("原始图像")

# 显示聚类图像
plt.subplot(1, 2, 2)
plt.imshow(dst, cmap='gray')
plt.title("K聚类图像")

plt.show()





