# 目标：制作一个直方图均衡化的函数接口，作为自制库，忽略参数校验的细节，仅实现基本功能
import numpy as np


def histogram(src):
    # 格式转换，确保转换为一维
    src_shape = src.shape
    src = src.reshape(src.size)
    # 遍历输入数据,计算每个像素出现的次数
    ni = np.zeros(256)
    for i in src:
        ni[i] += 1
    # print(ni)
    # 获取像素级中所有非0值
    tmp = ni.nonzero()
    # print(tmp)
    # 获取index最大的像素级
    src_m = np.max(tmp)
    # print(src_m)
    # 计算每个像素级的概率Pi
    pi = ni/src.size
    # print(pi)
    # 计算累加概率 sum_pi
    sum_pi = np.zeros(src_m + 1)
    sum_pi[0] = pi[0]
    for i in range(1, sum_pi.size):
        sum_pi[i] = sum_pi[i-1] + pi[i]
    # print(sum_pi)
    # 目标像素为 sum_pi *256 -1
    dst_pix = sum_pi * 256 - 1
    # print(dst_pix)
    # 四舍五入
    dst_pix = np.around(dst_pix)
    # print(dst_pix)
    # 替换像素点
    for i in range(src.size):
        src[i] = dst_pix[src[i]]
    # print(src)
    # np 变形
    src = src.reshape(src_shape)
    # print(src)
    return src


def main():
    # 源数据输入
    src = np.array([1, 3, 9, 9, 8, 2, 1, 3, 7, 3, 3, 6, 0, 6, 4, 6, 8, 2, 0, 5, 2, 9, 2, 6, 0])
    src = src.reshape(5, 5)
    print(src)
    dst = histogram(src)
    print(dst)


if __name__ == '__main__':
    main()

'''
if __name__ == '__ main__':
    main()
'''