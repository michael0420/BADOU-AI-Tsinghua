# -- coding:utf-8 --
import torch.nn as nn
import torch
import numpy as np
A=torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
B = A.flatten()
print(B.shape)
print(A.size())
C = A.view(2,-1)
print(C)
D = torch.LongTensor(4,1).random_()
print(D)
D = torch.tensor([[3],
              [1],
              [2],
              [3]])
F = torch.zeros(4,4).scatter_(0,D,1)
np.random.seed(1)
print(np.random.random())