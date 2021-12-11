from sklearn import datasets
import numpy as np

iris=datasets.load_iris()
data=iris['data']
target=iris['target']
a=1
pass

def cov_get(data,axis):
    assert axis in [0,1]
    d_shape=data.shape
    res_cov=np.zeros((d_shape[1-axis],d_shape[1-axis]))
    for i in range(d_shape[1-axis]):
        for j in range(d_shape[1-axis]):
            if axis==0:
                res_cov[i,j]=1/(d_shape[axis]-1)*np.dot(data[:,i]-np.mean(data[:,i]),data[:,j]-np.mean(data[:,j]))
    return res_cov
def pca_get(data:np.ndarray,num:int):
    """对数据进行pca降维

    Args:
        data (np.ndarray): 需要进行pca降维的数据
        num (int): pca降维后变量个数

    Returns:
        [type]: 返回可以进行pca降维后相对于原数据可解释的百分比，
        以及降维后按照重要性排序的变量，变量数与num对应
    """    
    center_data=data-np.mean(data,axis=0)
    ct_cov=np.cov(center_data,rowvar=False)
    # ct_cov1=cov_get(center_data,axis=0)
    eig,eig_mat=np.linalg.eig(ct_cov)
    sort_abseig=np.sort(np.abs(eig))[::-1]
    ep_perc=sum(sort_abseig[:num])/sum(sort_abseig)
    new_var=[]
    for i in range(num):
        idx=list(eig).index(sort_abseig[i])
        eig_var=eig_mat[:,idx]
        sig_var=np.dot(data,eig_var)
        new_var.append(sig_var)
    return ep_perc,new_var


if __name__=="__main__":
    pca_get(data,num=2)