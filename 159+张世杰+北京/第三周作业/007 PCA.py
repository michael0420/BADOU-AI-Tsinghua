# encoding: utf-8
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris  # 注意此处的base前要有横线

x, y = load_iris(return_X_y=True)  # 加载数据:x表示数据集中的属性数据，y表示数据标签;x 为列维度：各种属性的具体值； y 为行维度：鸢尾花种类
pca = dp.PCA(n_components=2)  # 加载PCA算法；dp:导入的包，ssklearn的一种方法：降维；PCA：用PCA算法（方法）；降到两维；降维后对应的y不变；
reduced_x = pca.fit_transform(x)
print(len(reduced_x))
print(reduced_x)
"""对产生的结果进行打印：首先创建三组空列表：这里的x,y都是画图中的坐标系的 x，y 和 上文的x，y无关， 注意单独的y 还是代表鸢尾花种类的"""
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)):  # 遍历降维后的数据集合：两个特征，150条数据，组成一个150个元素的序列；
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')  # plt.scatter函数的各个参数需要了解
plt.scatter(blue_x, blue_y, c='b', marker='*')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
