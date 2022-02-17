import numpy as np
import cv2

def infer_warpmatrix(src,dst):
    num=src.shape[0]
    A=np.zeros((2*num,8))
    B=np.zeros((2*num,1))
    for i in range(num):
        A_i=src[i,:]
        B_i=dst[i,:]
        A[2*i,:]=[A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        A[2*i+1,:]=[0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2*i,:]=B_i[0]
        B[2*i+1,:]=B_i[1]

    A=np.mat(A)   #生成矩阵.np.mat()为生成矩阵函数
    warpMatrix=A.I*B   #A.I为求A的逆矩阵，根据A*warpMatrix=B，得此公式

    warpMatrix=np.insert(warpMatrix,warpMatrix.shape[0],values=1,axis=0)   #插入a33=1，补足为9行1列矩阵
    warpMatrix=warpMatrix.reshape((3,3))   #生成3*3矩阵
    print(warpMatrix)
    return warpMatrix

# src=[[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
# src=np.array(src)
# dst=[[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
# dst=np.array(dst)
# infer_warpmatrix(src,dst)

"""透视变换"""
img=cv2.imread("photo1.jpg")
#获得原图像及新图像的四个顶点坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# m=cv2.getPerspectiveTransform(src,dst)  #调用库函数，生成透视变换矩阵
m=infer_warpmatrix(src,dst)               #调用变换矩阵函数，生成透视变换矩阵
new_img=cv2.warpPerspective(img,m,(337,488))  #调用库函数，生成新图像
cv2.imshow("src_img",img)
cv2.imshow("new_img",new_img)
cv2.waitKey(50000)
