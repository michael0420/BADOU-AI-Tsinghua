#coding =utf-8

import cv2
import numpy as np
import math
#import matplotlib.pylot as plt

def cannyDetail(lowThreshold):
	#1 高斯滤波
	sigma=0.5
	dim=np.int32(6*sigma+1)
	if dim%2==0:  #奇数大小的核
		dim=dim+1
	gaussian_filter=np.zeros((dim,dim),np.float32)
	tmp=[i-dim//2 for i in range(dim)]#生成高斯序列
	#计算系数
	n1=1/(2*math.pi*sigma**2)
	n2=-1/(2*sigma**2)
	for i in range(dim):
		for j in range(dim):
			gaussian_filter[i,j]=n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))

	gaussian_filter=gaussian_filter/gaussian_filter.sum()

	h,w=gray.shape
	filter_img=np.zeros((h,w),np.float32)
	dimw=dim//2
	filter_imgpad=np.pad(gray,((dimw,dimw),(dimw,dimw)),'constant')
	for i in range(h):
		for j in range(w):
			filter_img[i,j]=np.sum(filter_imgpad[i:i+dim,j:j+dim]*gaussian_filter)
	# plt.figure(1)
	# plt.imshow(filter_img.astype(np.uint8),cmap='gray')
	# plt.axis('off')

	#2 计算梯度
	sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	sobely=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	img_sobelx=np.zeros((h,w),dtype=np.float32)
	img_sobely=np.zeros((h,w),dtype=np.float32)
	sobelxy=np.zeros((h,w),dtype=np.float32)
	filter_imgpad=np.pad(filter_img,((1,1),(1,1)),'constant')
	for i in range(h):
		for j in range(w):
			img_sobelx[i,j]=np.sum(filter_imgpad[i:i+3,j:j+3]*sobelx)
			img_sobely[i,j]=np.sum(filter_imgpad[i:i+3,j:j+3]*sobely)
			sobelxy[i,j]=np.sqrt(img_sobelx[i,j]**2+img_sobely[i,j]**2)

	img_sobelx[img_sobelx==0]=0.00000001
	angle=img_sobely/img_sobelx
	# plt.figure(2)
	# plt.imshow(sobelxy.astype(np.uint8),cmap='gray')
	# plt.axis('off')
	#3 非极大值抑制，检测梯度方向是否为极大值
	img_nms=np.zeros((h,w),dtype=np.float32)
	for i in range(1,h-1):
		for j in range(1,w-1):
			flag=True # 8领域是否极大值的标记
			temp=sobelxy[i-1:i+2,j-1:j+2]#取8邻域梯度值
			if angle[i,j]<=-1:  #对梯度幅值进行线性差值，方向用angle，[90-135],temp(0,0)/(-k)+temp(0,1)*(1-k) temp(2, 1),temp(2,2)
				#g1  g2
				#     C
				#     g4 g3
				gradientL=(temp[0,1]-temp[0,0])/angle[i,j]+temp[0,1] #考虑梯度方向的正负值
				gradientR=(temp[2,1]-temp[2,2])/angle[i,j]+temp[2,1]
				if not(sobelxy[i,j]>gradientL and sobelxy[i,j]>gradientR):
					flag=False
			elif angle[i,j]>=1:#[45-90] 
				#     g2 g1
				#     C
				# g3  g4
				gradientL=(temp[0,2]-temp[0,1])/angle[i,j]+temp[0,1]
				gradientR=(temp[2,0]-temp[2,1])/angle[i,j]+temp[2,1]
				if not(sobelxy[i,j]>gradientL and sobelxy[i,j]>gradientR):
					flag=False
			elif angle[i,j]>0:#[0-45]  
				#        g1
				# g4  C  g2
				# g3
				gradientL=(temp[0,2]-temp[1,2])*angle[i,j]+temp[1,2]
				gradientR=(temp[2,0]-temp[1,0])*angle[i,j]+temp[1,0]
				if not(sobelxy[i,j]>gradientL and sobelxy[i,j]>gradientR):
					flag=False
			elif angle[i,j]<0:#[135-180] 
				# g3
				# g4  C  g2
				#        g1
				gradientL=(temp[1,0]-temp[0,0])*angle[i,j]+temp[1,0]
				gradientR=(temp[1,2]-temp[2,2])*angle[i,j]+temp[1,2]
				if not(sobelxy[i,j]>gradientL and sobelxy[i,j]>gradientR):
					flag=False
			if flag:
				img_nms[i,j]=sobelxy[i,j]

	# plt.figure(3)
	# plt.imshow(img_nms.astype(np.uint8),cmap='gray')

	#5 双阈值检测 连接线段
	edge=np.zeros((h,w),np.uint8)
	lower_boundary=lowThreshold
	high_boundary=lowThreshold*ratio
	xNum = [1, 1, 0, -1, -1, -1, 0, 1]  # 8邻域偏移坐标
	yNum = [0, 1, 1, 1, 0, -1, -1, -1]
	for i in range(1,h-1):
		for j in range(1,w-1):
			if(img_nms[i,j]>high_boundary):
				edge[i,j]=img_nms[i,j]
			elif(img_nms[i,j]>lower_boundary):
				for k in range(8):
					xx=j+xNum[k]
					yy=i+yNum[k]
					if(img_nms[yy,xx]>high_boundary):
						edge[i,j]=img_nms[yy,xx]
						break
	# plt.figure(4)
	# plt.imshow(edge.astype(np.uint8),cmap='gray')
	#
	# plt.axis('off')
	# plt.show()
	cv2.imshow('canny demo',edge.astype(np.uint8))	
		





if __name__=='__main__':

	lowThreshold=0
	highThreshold=100
	ratio=3
	kernel_size=3
	img=cv2.imread('lenna.png',cv2.IMREAD_UNCHANGED)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	h,w=gray.shape
	gray=cv2.resize(gray,(h//3,w//3))
	cv2.namedWindow('canny demo')
	cv2.createTrackbar('min threshold','canny demo',lowThreshold,highThreshold,cannyDetail)
	cannyDetail(0)
	if(cv2.waitKey(0)==27):
		cv2.destroyAllWindows()