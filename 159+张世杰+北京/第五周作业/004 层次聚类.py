# encoding: utf-8

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

'''dendrogram：系绘制树状图，linkage：层次聚类，fcluster:把聚类结果压缩到平面'''

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, 'ward')
f = fcluster(Z, 0.1, 'inconsistent') # inconsistent 表示为相关性（t的值为01之间），distance 绝对距离
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print('Z:\n',Z)
print('f:\n',f)
plt.show()
