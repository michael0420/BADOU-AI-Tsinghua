# -- coding:utf-8 --
import numpy as np
import cv2
import matplotlib.pyplot as plt

class WarpPerspectiveMatrix:
    def __init__(self,src,dst) :
        self.src = src
        self.dst = dst
        self.warpMatrix = self._WarpPerspectiveMatrix()

    def _WarpPerspectiveMatrix(self):
        assert self.src.shape[0] == self.dst.shape[0] and self.src.shape[0] >= 4
        nums = self.src.shape[0]
        A = np.zeros((2*nums, 8))
        B = np.zeros((2*nums, 1))
        for i in range(0,nums):
            src_i = self.src[i,:]
            dst_i = self.dst[i,:]
            A[2*i, :] = [src_i[0], src_i[1], 1, 0, 0, 0, -src_i[0]*dst_i[0], -src_i[1]*dst_i[0]]
            B[2*i] = dst_i[0]
            A[2*i+1,:] = [0,0,0,src_i[0],src_i[1],1,-src_i[0]*dst_i[1],-src_i[1]*dst_i[1]]
            B[2*i+1] = dst_i[1]
        A = np.mat(A) # 将A转换成矩阵
        warpMatrix = A.I * B # 矩阵的逆 * B
        warpMatrix = np.array(warpMatrix).T[0] # 转置后按第0维展开
        warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
        warpMatrix = warpMatrix.reshape((3, 3))
        return warpMatrix

    """
    变换矩阵
    [[ 6.90661007e-01  2.90965834e-01 -1.86502974e+02]
     [-8.97834002e-01  2.09717942e+00 -1.33817468e+02]
     [-9.11349961e-04  1.95386274e-03  1.00000000e+00]]
    """
    def transformImage(self,src_img,row,col):
        target_img = np.zeros((row,col,3),np.uint8)# 487 *314
        for i in range(src_img.shape[0]):# i是行 是高度
            for j in range(src_img.shape[1]):# j是列是宽度
                x,y,z = np.dot(self.warpMatrix,np.array([j,i,1]))
                x = int(x/z+0.5)
                y = int(y/z+0.5)
                if i == 601 and j == 17 :
                    print(x)
                    print(y)
                if x >=0 and x < col and y>=0 and y<row:
                    target_img[y,x] = src_img[i,j]
        return target_img

if __name__ == '__main__':
    img = cv2.imread("./img/photo1.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    src = [[207,151],[517,285],[17,601],[343,731]]
    src = np.array(src)
    dst = [[0,0],[337,0],[0,488],[337,488]]
    dst = np.array(dst)
    w = WarpPerspectiveMatrix(src,dst)
    result = w.transformImage(img,488,337)
    plt.imshow(result)
    plt.show()





