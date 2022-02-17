import numpy as np
import os
import cv2
import copy
def pca_fun(x, k):
    orix = copy.copy(x)
    k = k

    x = x - x.mean(axis=0)
    convx = np.dot(x.T,x)/(x.shape[0]-1)
    a, b = np.linalg.eig(convx)
    tig = np.argsort(-1*a)
    pca_array = np.asarray([b[:,tig[i]] for i in range(k)])
    pca_array = np.dot(orix,pca_array.T)

    return pca_array






if __name__=='__main__':

    img = cv2.imread('../lenna.png')
    gray_img = (img[:,:,0]*0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11).astype(np.uint8)
    img_fea = cv2.resize(gray_img,(10,8))

    pca_fea = pca_fun(img_fea,6)
    print(pca_fea)
    
