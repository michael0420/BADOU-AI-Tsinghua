import numpy as np
import cv2
import random

def CalculateModel(data):
    """
    假设model 为y = k*x+b
    根据数据计算k和b
    """

    n = data.shape[0]
    xn = data[:,0]
    yn = data[:,1]
    k_up = np.sum(xn*yn,axis=0)*n-np.sum(xn,axis=0)*np.sum(yn,axis=0) 
    k_down = np.sum(xn**2,axis=0)*n-np.sum(xn,axis=0)**2
    k = k_up/k_down
    b = np.sum(yn,axis=0)/n-k*np.sum(xn,axis=0)/n
    return k,b

    

def Ransac(datas,ransac_n,k,b,methods='a'):
    """
    iters = 40
    """
    datas = datas
    src_k = k
    src_b = b
    src_x = datas[:,0]
    src_y = datas[:,1]
    data_dimn = datas.shape[0]
    randomlist = [i for i in range(data_dimn)]
    if methods=='a':
        while True:
            data_sample = random.sample(randomlist,ransac_n)
            data = datas[data_sample]
            k, b = CalculateModel(data)
            y_ = k*src_x + b
            loss = (src_y - y_)**2
            loss_mask = loss<10
            if sum(loss_mask)>data_dimn*0.25:
                print('loss:',loss)
                print('lossmask:',loss_mask)
                print('sumloss:',sum(loss_mask))
                print('k,b:',k,b)
                print('src_k,src_b:',src_k,src_b)
                break


    elif methods=='b':
        while True:
            data_sample = random.sample(randomlist,ransac_n)
            data = datas[data_sample]
            k, b = CalculateModel(data)
            if k>=src_k-0.1 and k<=src_k+0.1:
                if b>=src_b-1 and b<=src_b+1:
                    break
        return k,b








def Main():
    """
    主函数

    """
    # 生成数据矩阵
    img = cv2.imread('../lenna.png')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    data = cv2.resize(gray_img,(2,20))
    k = np.random.random()*5
    b = np.random.random()*10
    x = random.sample(list(np.arange(-200,200,2)),30)
    x = np.array(x,dtype=np.int16)
    x = x.reshape((30,1))
    y = k*x + b
    xy = np.hstack((x,y))
    datas = np.vstack((xy,data))
    # ransac
    ransac_n = 10
    rasac_methods = 'a'
    Ransac(datas,ransac_n,k,b,rasac_methods)




if __name__=='__main__':
    Main()
