import cv2
import numpy as np

def sigmoid(x):
    y = 1/(1+np.exp(-1*x))
    return y

def forward_fun(x,w,b):
    """
    前向传播
    """
    w1, w2 = w
    zh = np.dot(x,w1) + b[0][0]
    ah = sigmoid(zh)
    zo = np.dot(ah,w2) + b[0][1]
    ao = sigmoid(zo)
    return zh,ah,zo,ao


def back_propagation(zh,ah,zo,ao,dst_o,w,x):
    """
    反向传播过程推导
    """
    # Etotal = 0.5*(target_o1-a_o1)**2+0.5*(target_o2-a_o2)**2
    # a_Etotal/a_a_o1 = -(target_o1-a_o1)
    # a_a_o1/w5 = a_o1*(1-a_o1)*h1
    # 求损失对w5，w6，w7,w8的偏导
    a = -(dst_o-ao)*ao*(1-ao)
    aw = np.vstack((a,a))*ah
    # 求损失对w1,w2,w3,w4的偏导
    w1, w2 = w
    b = np.sum(a*w2,axis=1)*ah*(1-ah)
    bw = np.vstack((b,b))*x
    
    return aw,bw

def handpush_network(a):
    """
    手推神经网路训练过程
    """
    dst_o = np.array([[0.01,0.99]],dtype=np.float32)
    # 生成输入x和网络参数w与b
    x = np.random.random((1,2))
    w1 = np.random.random((2,2))
    w2 = np.random.random((2,2))
    b = np.random.random((1,2))
    
    # 前向传播结果计算
    zh, ah, zo, ao = forward_fun(x,(w1,w2),b)
    print('zh:',zh,zh.shape)
    print('ah:',ah)
    print('zo:',zo)
    print('ao:',ao)
    print('dst_o:',dst_o)

    aw, bw = back_propagation(zh,ah,zo,ao,dst_o,(w1,w2),x)
    
    print('aw:',aw)
    print('bw:',bw)
    w1 = w1 - a*bw
    w2 = w2 - a*aw

    



if __name__=='__main__':
    a = 0.01 # 学习率
    handpush_network(a)
