import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1、读取数据
img = cv2.imread('lenna.png')
print(img.shape)

# 2、将二维数据转换为1维
data = img.reshape((-1, 3)) # -1表示我们不用亲自指定一维的大小，3表示RGB三个通道
data = np.float32(data)
print(data)

# 3、聚类
# 停止条件(type, max_inter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 初始类中心
flags = cv2.KMEANS_RANDOM_CENTERS

# 进行K-means聚类，聚成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

# 进行K-means聚类，聚成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 进行K-means聚类，聚成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)

# 进行K-means聚类，聚成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

# 进行K-means聚类，聚成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)


# 4、将图像转换为uint8
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]   # 使用每个簇的中心点作为颜色值，填充所有的聚类得到的结果，labels2里面保存这个每个像素点的
                                    # 标签，也就是这个像素点属于哪个簇，不同的簇其质心，也就是像素值不一样，保存在centers2里面
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

# 5、转换颜色

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)



# 6、显示图片
plt.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示中文
plt.figure(figsize=(6, 8), dpi=100)  # 画布10*10寸，dpi=100
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 子图横竖间隔0.3英寸

# 显示原始灰度图
plt.subplot(3, 2, 1)
plt.imshow(img)
plt.title("原始图像")

# 显示聚类图像
plt.subplot(3, 2, 2)
plt.imshow(dst2)
plt.title("聚类图像 K=2")

# 显示聚类图像
plt.subplot(3, 2, 3)
plt.imshow(dst4)
plt.title("聚类图像 K=4")


# 显示聚类图像
plt.subplot(3, 2, 4)
plt.imshow(dst8)
plt.title("聚类图像 K=8")


# 显示聚类图像
plt.subplot(3, 2, 5)
plt.imshow(dst16)
plt.title("聚类图像 K=16")

# 显示聚类图像
plt.subplot(3, 2, 6)
plt.imshow(dst64)
plt.title("聚类图像 K=64")

plt.show()
cv2.waitKey(0)
