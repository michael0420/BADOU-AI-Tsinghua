#coding:utf-8

import numpy as np
import  matplotlib.pyplot as plt

#Two ways of normalizaton
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x-x_min)/(x_max-x_min)'''
    x = [(float(i)-min(x))/(max(x)-min(x)) for i in x]
    return x

def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x-x_mean)/(x_max-x_min)'''
    x = [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]
    return x

# standardization
def z_score(x):
    '''x=(x-μ)/σ'''
    x_mean = np.mean(x)
    s2 = sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
li=[]
cs=[]
for i in l:
    c = l.count(i)      #统计某个元素在列表中出现的次数
    cs.append(c)
print(cs)

n = Normalization2(l)
z = z_score(l)
print(n)
print(z)
''''
蓝线为原始数据，橙线为z
plt.plot(x, y), x为x轴数据, y为y轴数据
'''
plt.plot(l, cs)
plt.plot(z, cs)
plt.show()