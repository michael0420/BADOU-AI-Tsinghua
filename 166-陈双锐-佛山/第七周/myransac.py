import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab

def mytest():
	data_num = 100
	x_exact = 30 * np.random.random((data_num,1))
	real_k = np.random.normal(10)
	y_exact = sp.dot(x_exact,real_k)
	x_noisy = x_exact + np.random.normal(size=x_exact.shape)
	y_noisy = y_exact + 10*np.random.normal(size=y_exact.shape)
	
	outlier_num = 20
	all_idx = np.arange(x_noisy.shape[0])
	np.random.shuffle(all_idx)
	out_idx = all_idx[:outlier_num]
	x_noisy[out_idx] = 30 * np.random.random(size=(outlier_num,1))
	y_noisy[out_idx] = 20 * np.random.normal(size=(outlier_num,1))
	
	all_data = np.hstack((x_noisy,y_noisy))
	model = LinearLeastSquareModel([0],[1],False)
	
	linear_k,resids,rank,s = sp.linalg.lstsq(all_data[:, range(1)], all_data[:, 1])
	ransac_k , ransac_idx = ransac(all_data, model, 10, 10)
	sort_idxs = np.argsort(x_exact[:,0])
	x_sort = x_exact[sort_idxs]
	pylab.plot(x_noisy[:,0], y_noisy[:,0],"k.", label = "data")
	pylab.plot(x_noisy[ransac_idx["inlier_idx"],0], y_noisy[ransac_idx["inlier_idx"],0],"bx", label = "ransac data")
	pylab.plot(x_sort,np.dot(x_sort, ransac_k),label = "ransac fit")
	pylab.plot(x_sort,np.dot(x_sort, real_k),label = "real fit")
	pylab.plot(x_sort,np.dot(x_sort, linear_k),label = "linear fit")
	pylab.legend()
	pylab.show()
	

def ransac(all_data, model, t_err, t_inlier):
	iteration = 0
	besterr = np.inf
	best_idx = None

	while iteration < 100:
		maybe_idx, test_idx = random_partition(10, all_data.shape[0])
		print("test_idx:", test_idx)
		maybe_data = all_data[maybe_idx,:]
		test_data = all_data[test_idx,:]
		maybemodel = model.fit(maybe_data)
		test_err = model.get_err(test_data, maybemodel)
		print("test_err:", test_err)
		also_idx = test_idx[test_err < t_err]
		print("also_idx:", also_idx)
		also_data = all_data[also_idx]

		if(len(also_data) > t_inlier):
			better_data = np.concatenate((maybe_data,also_data))
			better_model = model.fit(better_data)
			better_err = model.get_err(better_data,better_model)
			thiserr = np.mean(better_err)
			if (thiserr < besterr):
				besterr = thiserr
				bestfit = better_model
				best_idx = np.concatenate((maybe_idx, also_idx))

		iteration += 1
	if bestfit is None:
		raise ValueError("didn't meet fit acceptance criteria")
	else:
		return bestfit, {"inlier_idx" : best_idx}

def random_partition(n, n_data):
	all_idx = np.arange(n_data)
	np.random.shuffle(all_idx)
	maybe_idx = all_idx[:n]
	test_idx = all_idx[n:]
	return maybe_idx, test_idx
	

class LinearLeastSquareModel:
	# 最小二乘求线性解,用于RANSAC的输入模型
	def __init__(self, input_columns, output_columns, debug=False):
		self.input_columns = input_columns
		self.output_columns = output_columns
		self.debug = debug
	
	def fit(self, data):
		# np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
		A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
		B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
		x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
		return x  # 返回最小平方和向量
	
	def get_err(self, data, model):
		A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
		B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
		B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
		err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
		return err_per_point
	
		
if __name__ == "__main__":
	mytest()