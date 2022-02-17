"""
最小二乘法
"""
import pandas as pd

# 读入数据
data = pd.read_csv('../00-data/csv/least_squares_data.csv', engine='python')
print(data)
X = data['X'].values
Y = data['Y'].values

assert len(X) == len(Y)

# 变量初始化
sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0

# 计算中间变量的值
n = len(X)
for i in range(n):
    sum1 = sum1 + X[i]          # X[i]的累加和
    sum2 = sum2 + Y[i]          # Y[i]的累加和
    sum3 = sum3 + X[i] * X[i]   # X[i]**2 的累加和
    sum4 = sum4 + X[i] * Y[i]   # X[i]*Y[i] 的累加和

# 计算斜率k和截距b
k = (n * sum4 - sum1 * sum2) / (n * sum3 - sum1 * sum1)
b = (sum2 - k * sum1) / n
print("Coeff: {} Intercept: {}".format(k, b))
print('y = {}x + {}'.format(k, b))

