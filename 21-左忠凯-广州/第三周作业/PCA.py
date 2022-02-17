'''对鸢尾花进行降维'''
# sklearn自带鸢尾花数据介绍https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

x,y = load_iris(return_X_y=True) # 加载鸢尾花数据,x表示属性数据，y表示数据标签

print("x=\n{}".format(x))
print("y=\n{}".format(y))
print("样本数量:{}".format(x.shape[0]))
print("特征数量:{}".format(x.shape[1]))

pca = dp.PCA(n_components=2) # 降低到2维
reduced_x = pca.fit_transform(x) # 对原数据进行降维处理
print("降维后的矩阵:\n{}".format(reduced_x))

# 按鸢尾花的类别，将降维后的数据点保存在不同的表格上
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y =[], []

for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()






