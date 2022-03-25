# 目标：复现课程中的直方图均衡化过程（5*5手推变化）
import numpy as np

# 源数据输入
src = np.array([1, 3, 9, 9, 8, 2, 1, 3, 7, 3, 3, 6, 0, 6, 4, 6, 8, 2, 0, 5, 2, 9, 2, 6, 0])
print(src)

# 遍历输入数据,计算每个像素出现的次数
Ni = np.zeros(256)
for i in src:
    Ni[i] += 1
print(Ni)
# 获取像素级中所有非0值
tmp = Ni.nonzero()
print(tmp)
# 获取index最大的像素级
src_m = np.max(tmp)
print(src_m)
# 计算每个像素级的概率Pi
Pi = Ni/src.size
print(Pi)
# 计算累加概率 sumPi
sumPi = np.zeros(src_m + 1)
sumPi[0] = Pi[0]
for i in range(1, sumPi.size):
    sumPi[i] = sumPi[i-1] + Pi[i]
print(sumPi)
# 目标像素为 sumPi *256 -1
dst_pix = sumPi * 256 - 1
print(dst_pix)
# 四舍五入
dst_pix = np.around(dst_pix)
print(dst_pix)
# 替换像素点
for i in range(src.size):
    src[i] = dst_pix[src[i]]
print(src)
# np 变形
src = src.reshape(5, 5)
print(src)




