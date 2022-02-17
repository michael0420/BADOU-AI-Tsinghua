import numpy as np
import scipy.special


def calcLoss(a_out, target):
	loss = 0.0
	for i in range(len(target)):
		loss += 0.5 * (a_out[i] - target[i]) ** 2
	
	return sum(loss)


class NeuralNetwork:
	def __init__(self, n_input, n_hidden, n_output, lr):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output
		self.lr = lr
		self.wih = np.random.normal(0.0, pow(n_hidden, -0.5), (n_hidden, n_input))
		self.who = np.random.normal(0.0, pow(n_output, -0.5), (n_output, n_hidden))
		self.activate = lambda x: scipy.special.expit(x)
	
	def train(self, x, target):
		x = np.array(x, ndmin=2).T
		target = np.array(target, ndmin=2).T
		# 隐藏层输入
		z_hidden = np.dot(self.wih, x)
		# 隐藏层输出
		a_hidden = self.activate(z_hidden)
		# 输入层输入
		z_out = np.dot(self.who, a_hidden)
		# 输出层输出
		a_out = self.activate(z_out)
		
		# 损失
		loss = calcLoss(a_out, target)
		print("loss=", loss)
		# 输出层误差
		err_out = (a_out - target) * a_out * (1 - a_out)
		# 隐藏层误差
		err_hidden = np.dot(self.who.T, err_out) * a_hidden * (1 - a_hidden)
		# 误差反射传播，更新链路权重
		self.who -= self.lr * np.dot(err_out, np.transpose(a_hidden))
		self.wih -= self.lr * np.dot(err_hidden, np.transpose(x))
	
	def query(self, x):
		x = np.array(x,ndmin = 2).T
		z_hidden = np.dot(self.wih, x)
		a_hidden = self.activate(z_hidden)
		z_out = np.dot(self.who, a_hidden)
		a_out = self.activate(z_out)
		return np.around(np.transpose(a_out),4)

if __name__ == "__main__":
	n_input = 784
	n_hidden = 200
	n_output = 10
	lr = 0.1
	epoch = 10
	nn = NeuralNetwork(n_input, n_hidden, n_output, lr)
	f = open("dataset/mnist_train.csv", "r")
	xx = f.readlines()
	f.close()
	for j in range(epoch):
		for i in range(len(xx)):
			rowdata = xx[i].split(",")
			x = rowdata[1:]
			x = np.asfarray(x) / 255.0 * 0.99 + 0.01
			label = int(rowdata[0])
			target = np.zeros(n_output) + 0.01
			target[label] = 0.99
			nn.train(x, target)
		
	f = open("dataset/mnist_test.csv", "r")
	xx = f.readlines()
	f.close()
	score = []
	for i in range(len(xx)):
		rowdata = xx[i].split(",")
		x = np.asfarray(rowdata[1:]) / 255.0 + 0.01
		y = nn.query(x)
		print(y)
		label = np.argmax(y)
		print("该图片对应的数字为：", rowdata[0])
		print("网络认为图片数字为：", label)
		print("----------------------")
		if (label == int(rowdata[0])):
			score.append(1)
		else:
			score.append(0)
	
	score = np.asarray(score)
	print("准确率：", score.sum() / score.size)

