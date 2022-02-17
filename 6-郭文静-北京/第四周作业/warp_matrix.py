#coding=utf-8

'''
透视变换实现
'''
import cv2
import numpy as np

def  calc_warp_matrix(srcPoint,dstPoint):
	assert srcPoint.shape[0]==dstPoint.shape[0] and srcPoint.shape[0]>=4
	nums=srcPoint.shape[0]
	# A*warpMatrix=B
	A=np.zeros((2*nums,8),np.float32)
	B=np.zeros((2*nums,1),np.float32)
	# a11*x+a12*y+a13-a31*x*X-a32*y*X=X
	# a21*x+a22*y+a23-a31*x*Y-a32*y*Y=Y
	for i in range(nums):
		A_i=srcPoint[i,:]
		B_i=dstPoint[i,:]
		A[2*i,:]=[A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
		A[2*i+1,:]=[0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
		B[2*i]=B_i[0]
		B[2*i+1]=B_i[1]
	A =np.mat(A)
	warpMatrix=A.I*B
	warpMatrix=np.array(warpMatrix).T[0]
	warpMatrix=np.insert(warpMatrix,warpMatrix.shape[0],values=1.0,axis=0)#插入a33=1
	warpMatrix=warpMatrix.reshape(3,3)
	return warpMatrix



def warp_img(srcImg,warpMatrix):
	h,w,c=srcImg.shape
	dstImg=np.zeros((h,w,c),srcImg.dtype)
	for i in range(h):
		for j in range(w):
			dx=warpMatrix[0,0]*j+warpMatrix[0,1]*i+warpMatrix[0,2]
			dy=warpMatrix[1,0]*j+warpMatrix[1,1]*i+warpMatrix[1,2]
			dz=warpMatrix[2,0]*j+warpMatrix[2,1]*i+warpMatrix[2,2]+0.00000001
			dx=np.int32(dx/dz+0.5)
			dy=np.int32(dy/dz+0.5)
			if(dx>0 and dy >0 and dx < w and dy < h):
			 	dstImg[i,j,:]=srcImg[dy,dx,:]

	warpImg=dstImg[0:488,0:337]
	return warpImg


if __name__=='__main__':
	srcImg=cv2.imread('photo1.jpg',cv2.IMREAD_UNCHANGED)
	srcPoint=np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
	dstPoint=np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
	warpMatrix=calc_warp_matrix(dstPoint,srcPoint)
	warpImg=warp_img(srcImg,warpMatrix)
	cv2.imshow('srcImg',srcImg)
	cv2.imshow('warpImg',warpImg)
	cv2.waitKey(0)

