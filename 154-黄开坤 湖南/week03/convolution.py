#coding:utf-8

import numpy as np


'''
Convolution code implementation
matrix:[h, w]; kernel:[k, k]
batch_size=1,padding='SAME'
'''
input_data=([
              [[1,0,1,2,1],
               [0,2,1,0,1],
               [1,1,0,2,0],
               [2,2,1,1,0],
               [2,0,1,2,0]],

               [[2,0,2,1,1],
                [0,1,0,0,2],
                [1,0,0,2,1],
                [1,1,2,1,0],
                [1,0,1,1,1]],

                [[1,2,1,3,0],
                 [2,0,1,2,1],
                 [1,0,1,2,1],
                 [0,2,1,1,2],
                 [2,1,0,1,1]]
            ])
weights_data1=([
               [[ 1, 0, 1],
                [-1, 1, 0],
                [ 0,-1, 0]],

               [[-1, 0, 1],
                [ 0, 0, 1],
                [ 1, 1, 1]],

               [[ 1, 0,-1],
                [-1, 0, 1],
                [ 1, 0, 1]]
           ])
weights_data2=([
               [[-1, 0, 1],
                [ 1,-1, 0],
                [ 0,-1, 1]],

               [[-1, 0, 1],
                [ 0, 0, 1],
                [ 1,-1, 1]],

               [[ 1, 0,-1],
                [-1, 0, 1],
                [ 1, 0, -1]]
           ])

#窗口滑动计算的结果
# return res:[h,w], stride=1
def computer_conv(matr, kernel):
    [h, w] = matr.shape
    [k, _] = kernel.shape
    r = int(k/2)    #半经减一
    #定义边界填充0后的map
    padding_matr = np.zeros((h+2, w+2), dtype=np.float32)
    #用于保存最后de计算结果
    result = np.zeros((h, w), dtype=np.float32)
    #中间区域赋值输入矩阵
    padding_matr[1: h+1, 1: w+1] = matr
    #遍历每个中心点
    for y in range(1, h+1):
        for x in range(1, w+1):
            #取出当前覆盖的k*k矩阵
            roll = padding_matr[y-r: y+r+1, x-r: x+r+1]
            #calculation convolution
            result[y-1][x-1] = np.sum(roll * kernel)
    return result     #(5, 5)

#窗口滑动计算的结果
# return result:[3,3], stride=2
# def computer_conv2(matr, kernel):
#     [h, w] = matr.shape
#     [k, _] = kernel.shape
#     r = int(k/2)    #半经减一
#     #定义边界填充0后的map
#     padding_matr = np.zeros((h+2, w+2), dtype=np.float32)
#     #用于保存最后de计算结果
#     result = np.zeros((3, 3), dtype=np.float32) # k = ((h-k)/s + 1)
#     #中间区域赋值输入矩阵
#     padding_matr[1: h+1, 1: w+1] = matr
#     #遍历每隔一个中心点
#     for y in range(1, h+1, 2):
#         for x in range(1, w+1, 2):
#             #取出当前覆盖的k*k矩阵
#             roll = padding_matr[y-r: y+r+1, x-r: x+r+1]
#             #calculation convolution
#             result[int((y-1)/2)][int((x-1)/2)] = np.sum(roll * kernel)
#     return result   #(3, 3)


#对应通道相乘再相加
def my_conv2d(input, weight):
    [c, h, w] = input.shape
    # k = weight.shape[0]
    outputs = np.zeros([h, w], np.float32)  #stride=1
    # outputs = np.zeros([3, 3], np.float32)  #stride=2

    # traveral each channel遍历
    for i in range(c):
        #feature map -->[h,w]
        f_mat = input[i]
        #kernel -->[k, k]
        w = weight[i]
        result = computer_conv(f_mat, w)  #stride=1
        # result = computer_conv2(f_mat, w)   #stride=2
        outputs = outputs + result
    return outputs

def main():
    #shape = [c, h, w]
    input = np.asarray(input_data, np.float32)  #Convert the input to an array
    w1 = np.asarray(weights_data1, np.float32)
    w2 = np.asarray(weights_data2, np.float32)
    w = (w1, w2)
    for i in range(len(w)):
        result = my_conv2d(input, w[i])
        print(result,'->')

if __name__ == '__main__':
    main()


