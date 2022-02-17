import cv2
import numpy as np

class pca():
	def __init__(self,x):
		self.x = x
		self.k = 2
		self.centerx = self.centrarize()
		self.cov = self._cov()
		self.u = self._u()
		self.z = self._z()
	def centrarize(self):
		print("最初的x\n", x)
		mean = [np.mean(e) for e in self.x.T]
		print("平均值mean\n", mean)
		centerx = self.x - mean
		print("中心化后的centerx\n", centerx)
		return centerx
		
	def _cov(self):
		ns = np.shape(self.x)[0]
		cov = np.dot(self.centerx.T, self.centerx) / (ns - 1)
		print("协方差：\n", cov)
		return cov
	
	def _u(self):
		a,b = np.linalg.eig(self.cov)
		print("特征值：\n", a)
		#形式为(e1,e2,e3),每个特征向量均为列向量
		print("特征向量：\n", b)
		idx = np.argsort(-1 * a)
		print("idx：\n", idx)
		uT = [b[:, idx[i]] for i in range(self.k)]
		print("转换矩阵1：\n", uT)
		u = np.transpose(uT)
		print("转换矩阵2：\n", u)
		return u
	
	def _z(self):
		z = np.dot(self.x, self.u)
		print("降维后的样本：\n",z)
		return z.T


if __name__ == "__main__":
	x = np.array([[63,62,73],
					[47,77,8],
					[48,46,38],
					[50,48,0],
					[68,86,75],
					[64,7,33],
					[34,70,56],
					[48,27,55],
					[44,21,94],
					[55,42,30]])
	pca(x)

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
