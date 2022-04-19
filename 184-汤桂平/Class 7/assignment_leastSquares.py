# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/4/3

import pandas as pd
import numpy as np

'''
完全根据最小二乘法原理及计算步骤即可实现以下代码
即根据最小残差平方和（其他求范式也可以）求得的即为最佳拟合函数，因此根据多元函数极值定理可知只需要
用残差平方和对系数k和b求偏导，然后令偏导数等于零得到两个等式。然后求出k和b；
'''
data = pd.read_csv('train_data.csv')  # 利用pandas处理csv文档
X = data['X']
Y = data['Y']

s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = np.size(X)

for i in range(n):   # 把求k和b的公式进行代码化
    s1 = s1 + X[i] * Y[i]
    s2 = s2 + X[i]
    s3 = s3 + Y[i]
    s4 = s4 + X[i]**2
k = (n * s1 - s2 * s3) / (n * s4 - s2**2)
b = s3 / n - k * s2 / n

print("Coeff: {}   Intercept: {}".format(k, b))