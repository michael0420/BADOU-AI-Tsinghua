import numpy as np

def cen(X):
    x = X
    m = np.array([np.mean(i) for i in x.T])
    x_cen = x - m
    return x_cen

def cov(x_cen):
    x = x_cen
    n = np.shape(x)[0]
    c = np.dot(x.T,x)/(n-1)#不同维度间的相关性
    return c

def pca(X,c,K):
    x = X
    cov = c
    k = K
    vec = []
    eig_val, eig_vec = np.linalg.eig(cov)#计算协方差矩阵cov的特征值，特征向量
    ind = np.argsort(-1 * eig_val)#根据特征值大小，从大到小进行排序，返回对应的index
    for i in range(k):#取前k个特征向量
        vec.append(eig_vec[i])
    v = np.transpose(vec)
    x_k = np.dot(x,v)
    return x_k

if __name__=='__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    x_cen = cen(X)
    c = cov(x_cen)
    x_k = pca(X,c,K)
    print("X:",X)
    print(np.shape(X))
    print("x_k:",x_k)
    print(np.shape(x_k))

