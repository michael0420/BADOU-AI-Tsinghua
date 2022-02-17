from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

def data_generate(num_list=(500,500)):
    num1,num2=num_list
    X=np.linspace(-1,3,num=num1)
    y_exact=5*X+1
    #TODO to parameterize
    mu=0
    sigma=0.1
    np.random.seed(0)
    noise=np.random.normal(mu,sigma,(num1,))
    y=y_exact+noise[0:]
    ##TODO to parameterize
    cord=(1,4)
    sigma1=np.array([[3,0],[0,3]])
    noise_batch=np.random.multivariate_normal(cord,sigma1,(num2,))
    X_end=list(X)+list(noise_batch[:,0])
    y_end=list(y)+list(noise_batch[:,1])
    return X_end,y_end

def data_visual(X,y):
    figure1=plt.figure()
    plt.scatter(X,y)
    ax=plt.gca()
    # locator=plt.MultipleLocator(1)
    # ax.xaxis.set_major_locator(locator)
    # ax.yaxis.set_major_locator(locator)
    # plt.xlim(-1,3)
    # plt.ylim(-4,16) 
    plt.show()

def result_visual(X,y,para):
    figure1=plt.figure()
    plt.scatter(X,y)
    plt.plot([min(X),max(X)],[min(X)*para[0]+para[1],max(X)*para[0]+para[1]],c='r')
    plt.show()
def ransac(X,y,k=500):
    n=int(len(X)/5)
    X,y=np.array(X),np.array(y)
    iter_num=0
    result_list=[]
    para_list=[]
    while iter_num<k:
        np.random.seed(iter_num)
        id_rand=np.random.choice(range(len(X)),n,replace=False)
        X_e,y_e=X[id_rand],y[id_rand]
        k_e,b_e=LSM_es(X_e,y_e)
        in_num,out_num=inout_nb_cp(getmodel(para=(k_e,b_e)),X,y)
        iter_num+=1
        result_list.append(in_num)
        para_list.append((k_e,b_e))
    max_innum=max(result_list)
    for i in range(k):
        if result_list[i]==max_innum:
            print(result_list[i],para_list[i])
    return para_list[result_list.index(max_innum)]
    # pass

def getmodel(para,kind="linear"):
    if kind=="linear":
        return lambda x:x*para[0]+para[1]

def inout_nb_cp(model,X,y):
    in_num,ot_num=0,0
    for id in range(len(X)):
        y_p=model(X[id])
        if abs(y[id]-y_p)/abs(y[id]+1e-7)<0.1:
            in_num+=1
        else:
            ot_num+=1
    return in_num,ot_num
def LSM_es(X,y):
    X,y=np.array(X),np.array(y)
    X_m,y_m=np.mean(X),np.mean(y)
    k=(np.mean(X*y)-X_m*y_m)/\
        (np.mean(X**2)-X_m**2)
    b=y_m-k*X_m
    return k,b
    
if __name__=="__main__":
    X,y=data_generate(num_list=(500,100))
    para=ransac(X,y)
    result_visual(X,y,para)