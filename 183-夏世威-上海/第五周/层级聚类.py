from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
'''
这里直接就出结果了
算类簇与类簇之间距离的时候:
single: 看最近的
complete:看最远的
ward:离差平方和
'''
Z = linkage(X, 'single')

# show
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()
