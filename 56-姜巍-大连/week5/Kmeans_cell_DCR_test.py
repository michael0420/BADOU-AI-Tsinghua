# coding=utf-8
from sklearn.cluster import KMeans

"""
第一部分：数据集
X表示二维矩阵数据，电芯在两种不同测试条件下的数据
总共30行，每行两列数据
第一列表示1C放电条件下的直流内阻DCR：DCR_1C
第二列表示2C放电条件下的直流内阻DCR：DCR_2C
注：原数据经过验证，产品实际在性能上可被分成3个档次：良品、次品和坏品，而外观上没有对区分这一性能有帮助的显著的差异。
"""
test_data = [[456.1, 484.15],
             [424.39, 453.66],
             [446.34, 475.61],
             [429.37, 450.0],
             [414.63, 436.59],
             [443.9, 475.61],
             [446.34, 476.83],
             [419.51, 439.02],
             [448.78, 482.93],
             [436.59, 474.39],
             [421.95, 431.71],
             [436.59, 451.22],
             [431.71, 441.46],
             [412.2, 420.73],
             [409.76, 418.29],
             [431.71, 443.9],
             [424.39, 435.37],
             [424.39, 437.8],
             [431.71, 442.68],
             [419.51, 426.83],
             [302.44, 331.71],
             [300.0, 328.05],
             [297.56, 323.17],
             [307.32, 335.37],
             [309.76, 336.59],
             [300.0, 326.83],
             [292.68, 318.29],
             [295.12, 318.29],
             [297.56, 324.39],
             [304.88, 332.93]
             ]

# 输出数据集
print(test_data)

"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(test_data) 载入数据集test_data，并且将聚类的结果赋值给y_pred
"""

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(test_data)

# 输出完整Kmeans函数，包括很多省略参数
print(clf)
# 输出聚类预测结果
print("y_pred = ", y_pred)

"""
第三部分：可视化绘图
"""

import numpy as np
import matplotlib.pyplot as plt

# 获取数据集的第一列和第二列数据 使用列表生成器 n[0]表示test_data第一列
x = [n[0] for n in test_data]
print(x)
y = [n[1] for n in test_data]
print(y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans DCR test Data")

# 绘制x轴和y轴坐标
plt.xlabel("DCR_1C")
plt.ylabel("DCR_2C")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()
