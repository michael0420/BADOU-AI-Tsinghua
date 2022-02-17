#coding = utf-8
import numpy as np
import pandas as pd
import pylab

def ransac(data, model, K, t ,inliers_num,):
    #随机选取两个点
    n=2
    iterations=0
    bestfit=None
    besterr=np.inf
    best_inlier_idxs=None
    while iterations < K:
        train_idxs,test_idxs= random_partition(n,data.shape[0])
        #计算模型
        train_points=data[train_idxs,:]
        test_points=data[test_idxs,:]
        trainmodel=model.fit(train_points)
        test_err=model.get_error(test_points,trainmodel)
        print('test_err=', test_err < t)
        #计算内点
        inliers_idx=test_idxs[test_err<t]
        print('inliers_idx= ', inliers_idx)
        inliers_points=data[inliers_idx,:]
        
        #更新最优值
        if(len(inliers_points)>inliers_num):
            betterdata=np.concatenate((train_points, inliers_points))
            #用最小二乘计算当前内点拟合结果
            bettermodel=model.fit(betterdata)
            better_errs=model.get_error(betterdata,bettermodel)
            thiserr=np.mean(better_errs)
            if thiserr <  besterr:
                besterr = thiserr
                bestfit= bettermodel
                best_inlier_idxs=np.concatenate((train_idxs, inliers_idx))
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteraia")
    else:
        return bestfit,{'inliers':best_inlier_idxs}
        
def random_partition(n,n_data):
    all_idxs=np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1=all_idxs[:n]
    idxs2=all_idxs[n:]
    return idxs1, idxs2
class LinearLeastSquareModel:
    def __init__(self,input_colums, output_colums, debug= False):
        self.input_colums = input_colums
        self.output_colums = output_colums
        self.debug = debug
        
    def fit(self, data):
        X=np.vstack([data[:,i] for i in self.input_colums]).T
        Y=np.vstack([data[:,i] for i in self.output_colums]).T
         #初始化赋值
        s1 = 0     
        s2 = 0
        s3 = 0     
        s4 = 0         
        n=len(X)
        #循环累加
        for i in range(n):
            s1 = s1 + X[i]*Y[i]     #X*Y，求和
            s2 = s2 + X[i]          #X的和
            s3 = s3 + Y[i]          #Y的和
            s4 = s4 + X[i]*X[i]     #X**2，求和

        #计算斜率和截距
        k = (s2*s3-n*s1)/(s2*s2-s4*n)
        b = (s3 - k*s2)/n
        return k,b
    
       
    def get_error(self,data,model):
        X=np.vstack([data[:,i] for i in self.input_colums]).T
        Y=np.vstack([data[:,i] for i in self.output_colums]).T
        result=X*model[0]+model[1]
        err_per_point=np.sum((Y-result)**2,axis=1)
        return err_per_point
    


 
data=pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')
X=data['X'].values
X=X.reshape(X.shape[0],1)
Y=data['Y'].values
Y=Y.reshape(Y.shape[0],1)
n_inputs=1
n_outputs=1
input_colums=list(range(n_inputs))
output_colums=list([n_inputs + i  for i in range(n_outputs)])
data=np.hstack([X, Y])
model=LinearLeastSquareModel(input_colums, output_colums, False)
k,b=model.fit(data)
print('k=%f b=%f'%(k,b))
iterations=1000
thresh=7
inliers_num=1

bestfit,ransac_inliers=ransac(data,model,iterations,thresh,inliers_num)
print(bestfit)
pylab.plot(X,Y,'k.',label='data')
pylab.plot(data[ransac_inliers['inliers'],0],data[ransac_inliers['inliers'],1],'bx',label='ransac data')
result=data[:,0]*bestfit[0]+bestfit[1]
pylab.plot(data[:,0],result,label='ransac fit')
pylab.legend()
pylab.show()
