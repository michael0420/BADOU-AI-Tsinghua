#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""最小二乘"""

def LeastSqaureMethod(data_x,data_y):
    if len(data_x)!=len(data_y):
        # 数据无效
        return None,None
    size_N = len(data_x)
    tmp_xy = 0
    tmp_x = 0
    tmp_y = 0
    tmp_x_square = 0
    for i in range(size_N):
        tmp_xy += data_x[i]*data_y[i]
        tmp_x += data_x[i]
        tmp_y += data_y[i]
        tmp_x_square +=data_x[i]*data_x[i]

    k = (size_N*tmp_xy-tmp_x*tmp_y)/(size_N*tmp_x_square-tmp_x*tmp_x)
    b = (tmp_y/size_N)-k*tmp_x/size_N
    return k,b


if __name__ == '__main__':
    data_x = [1,1.5,2,2.5,3,3.5,4,4.5]
    data_y = [2.1,2.8,3.8,4.9,5.5,7.5,8.6,8.8]
    k,b = LeastSqaureMethod(data_x,data_y)
    print(k)
    print(b)