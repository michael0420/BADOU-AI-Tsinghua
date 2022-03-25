#coding:utf-8

import pandas as pd

'''
https://www.cnblogs.com/wyy1480/p/10322336.html
'''

file_path = 'train_data.csv'
sales = pd.read_csv(file_path, sep='\s*,\s*', engine='python')
X = sales['X'].values
Y = sales['Y'].values
print(sales)

#初始化赋值
s1, s2, s3, s4, n = 0, 0, 0, 0, 4

#根据公式计算各求和
for i in range(n):
    s1 = s1 + X[i] * Y[i]
    s2 = s2 + X[i]
    s3 = s3 + Y[i]
    s4 = s4 + X[i]**2
#带入公式
k = (n*s1 - s2*s3)/(n*s4 - s2**2)
b = (s3-k*s2) / n
print('Coeff: {} Intercept: {}'.format(k,b))
