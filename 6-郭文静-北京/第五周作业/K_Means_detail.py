# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
#计算欧式距离
def dist_eclud(A, B):
	return np.sqrt(np.sum(np.square(A-B)))

#随机取K个中心
def rand_centers(data, K):
	centers = np.random.randint(255,size = (1, K))
	centers_data = data[centers, :][0]
	return centers_data

#聚类
def k_means(data, centers, K):
	m, n = data.shape
	cluster_dist = np.zeros((m, 2), dtype = np.float32)
	cluster_changed = True
	clusters=centers
	while cluster_changed:
		cluster_changed = False
		for i in range(m):
			mindist = np.inf
			minidx = -1
			#计算每个数据所属的类
			for j in range(K):
				dist=dist_eclud(clusters[j], data[i])
				if dist < mindist:
					mindist = dist
					minidx=j
				#质心是否发生变化
			if(cluster_dist[i, 0] != minidx):
				cluster_changed = True
			cluster_dist[i, :] = minidx, mindist**2
		#计算新类的质心
		#new_clusters = np.zeros((K,n), dtpye = np.float32)
		for i in range(K):
			indx  =  np.nonzero(cluster_dist[:,0] == i)[0]
			clusteri = data[indx]
			clusters[i,:] = np.mean(clusteri, axis=0)
		print(clusters)

	cluster_dist=np.uint8(cluster_dist)
	data_result=clusters[cluster_dist[:, 0], :]
	data_result=np.uint8(data_result)
	return clusters, cluster_dist, data_result

if __name__ == '__main__':
	#读取原始图像
	img  =  cv2.imread('lenna.png',cv2.IMREAD_COLOR)
	print (img.shape)
	data = img.reshape((-1,3))
	data=np.float32(data)
	K = 2
	centers = rand_centers(data, K)
	clusters, cluster_dist, data_result = k_means(data, centers, K)
	data_show=data_result.reshape(img.shape)

	img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	data_show=cv2.cvtColor(data_show,cv2.COLOR_BGR2RGB)

	#显示中文标签
	plt.rcParams['font.sans-serif']=['SimHei']
	titles = [u'原始图像', u'聚类图像 K = 2']
	images=[img,data_show]
	for i in range(2):
		plt.subplot(2,2,i+1)
		plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])
	plt.show()


