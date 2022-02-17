'''
@auther:Jelly
用于实现PCA算法

相关接口：
def PCA_self(Data,param_num)
    参数：
        Data：需要降维的数据
        param_num：降维后的参数个数
'''

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

def PCA_self(Data,param_num):
    '''
    函数用途：进行PCA降维
    参数：
        Data：需要降维的数据
        param_num：降维后的参数个数
    '''
    Data_x = np.array(Data)
    h,w = Data_x.shape

    # 数据零均值化 （中心化）
    mean = np.zeros(w)
    for i in range(w):
        mean[i] = np.mean(Data_x[:,i])
    Data_x = Data_x - mean

    # 求解数据协方差
    cov_Data = np.dot(np.transpose(Data_x),Data_x)/(h)

    # 求解协方差矩阵的 特征值 与 特征向量
    eigenvalue,featurevector = np.linalg.eig(cov_Data)

    # 按特征值大小选取所需要个数的特征值
    h_feat,w_feat = featurevector.shape
    featurevector_choice = np.zeros((h_feat,param_num))  #新建一个空的矩阵用于存储选择特征向量后的特征矩阵
    eig_order = np.argsort(-eigenvalue)

    for i in range(param_num):
        featurevector_choice[:,i] = featurevector[:,eig_order[i]]

    # 将数据集Data_x投射到选取的特征向量上，得到降维后的数据集
    Data_PCA = np.dot(Data_x,featurevector_choice)

    return Data_PCA



data_x,data_y = load_iris(return_X_y=True)

x_PCA = PCA_self(data_x,2)

print(x_PCA)

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]

for i in range(len(x_PCA)): #沿着列展开
    if data_y[i]==0:
        red_x.append(x_PCA[i][0])
        red_y.append(x_PCA[i][1])
    elif data_y[i]==1:
        blue_x.append(x_PCA[i][0])
        blue_y.append(x_PCA[i][1])
    else:
        green_x.append(x_PCA[i][0])
        green_y.append(x_PCA[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
