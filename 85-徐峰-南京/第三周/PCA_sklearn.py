#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/6 23:51 
# ide： PyCharm


import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
pca = PCA(n_components=2)
# pca.fit(X) #训练
newx = pca.fit_transform(X) #降维后的数据
print(pca.explained_variance_ratio_) #输出贡献率
print(newx)