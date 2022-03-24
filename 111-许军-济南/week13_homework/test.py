# -- coding:utf-8 --
import numpy as np
# permute reshpae view transpose 函数的区别

import torch
a = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
b = a.view(4,3) # view函数是先将tensor拉平，然后重新reshape成新形状为(4.3)的张量
c = a.permute(0,2,1)# 交换多个维度顺序
d = a.transpose(0,1)#交换两个维度顺序
print(d.size())
# reshape是numpy模块里的，调整形状
#有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。


# np.meshgrid 函数
e = np.array([1,2,3])
f = np.array([8,9,10])
shiftx,shifty = np.meshgrid(e,f) # 生成坐标点使用

#print(shiftx.ravel())# 拉成一维的数组

result = np.stack((shiftx,shifty,shiftx,shifty),axis=1)

#
x = torch.Tensor([[1,2,3,4],[1,3,4,6]])
print(x[:,0::4])
x = torch.unsqueeze(x,-1)# 在最后添加一个维度
print(x)

a = "djfajflf,ajfa,fjkaf, 8,0,0\n"
print(a.split())
