import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

features, labels = load_iris(return_X_y=True)
pca = dp.PCA(n_components=2)  # 根据特征值大小,取两个特征向量构成特征矩阵
results = pca.fit_transform(features)

result_x, result_y = [], []
for result in results:
    result_x.append(result[0])
    result_y.append(result[1])

# 不知道label的情况下看一下特征分布
plt.scatter(result_x, result_y, c='r')
plt.show()
