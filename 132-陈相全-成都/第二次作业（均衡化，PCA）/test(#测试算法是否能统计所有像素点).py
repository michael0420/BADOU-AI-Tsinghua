#测试算法是否能统计所有像素点
import numpy as np
img = np.array([[3,5,7,2],
                [2,6,2,6],
                [8,4,3,2],
                [4,8,1,7]])
num = img.flatten()
print(type(num))
print(num)
num = num.tolist()
print(type(num))

count = [0 for i in range(0,256)]#不写这个的话是空列表，就是错的，会报超出范围
for i in range(0,256):
        count[i] += num.count(i)
print(count)