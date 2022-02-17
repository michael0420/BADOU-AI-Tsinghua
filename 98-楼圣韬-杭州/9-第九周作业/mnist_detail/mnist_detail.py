from typing import Any
import numpy as np
from tqdm import tqdm
eps = 1e-8


class NeuralNetworkModule:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:   # Any 为任意类型
        return self.__getattribute__("forward")(*args, **kwds)

    def backward(self, loss, lr=None):
        pass


class Dense(NeuralNetworkModule):
    def __init__(self, input_channel, output_channel) -> None:
        self.weights = np.random.normal(
            0.0, pow(input_channel, -0.5), (output_channel, input_channel))
        self.bias = np.random.normal(
            0.0, pow(input_channel, -0.5), output_channel)

    def forward(self, x: np.ndarray):
        self.input_data = x
        out = np.einsum('ij,bj->bi', self.weights,
                        self.input_data, optimize=True) + self.bias
        return out

    def backward(self, loss, lr):
        m = loss.shape[1]
        loss_next = np.einsum('ji,bj->bi', self.weights,
                              loss, optimize=True)
        dw = np.einsum('bi,bk->ik', loss, self.input_data,
                       optimize=True)/(m*len(loss))
        db = np.einsum('bi->i', loss)/len(loss)

        self.weights -= lr*dw
        self.bias -= lr*db

        return loss_next


class Relu(NeuralNetworkModule):
    def forward(self, x: np.ndarray):
        self.out = np.maximum(0, x)
        return self.out

    def backward(self, loss, lr=None):
        loss = loss.copy()
        loss[self.out == 0] = 0
        return loss


class Sigmoid(NeuralNetworkModule):
    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def backward(self, x):
        return x*self.out*(1-self.out)


class Softmax(NeuralNetworkModule):
    def forward(self, x):
        v = np.exp(x - x.max(axis=-1, keepdims=True))
        self.a = v / (v.sum(axis=-1, keepdims=True)+eps)
        return self.a

    def backward(self, y):
        return self.a * (y - np.einsum('ij,ij->i', y, self.a, optimize=True))


class CrossEntropyLoss(NeuralNetworkModule):
    def __init__(self):
        # 内置一个softmax作为分类器
        self.classifier = Softmax()

    def backward(self):
        return self.classifier.a - self.y

    def forward(self, a, y):
        '''
        a: 批量的样本输出
        y: 批量的样本真值
        return: 该批样本的平均损失
        '''
        a = self.classifier.forward(a)   # 得到各个类别的概率
        self.y = y
        # 等价于 y* np.log(a+eps)
        loss = np.einsum('ij,ij->', y, np.log(a+eps),
                         optimize=True) / y.shape[0]
        return -loss


class Model(NeuralNetworkModule):
    def __init__(self) -> None:
        self.fc1 = Dense(784, 256)
        self.f = Relu()
        self.fc2 = Dense(256, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.f(out)
        out = self.fc2(out)

        return out

    def backward(self, loss, lr):
        loss = self.fc2.backward(loss, lr)
        loss = self.f.backward(loss)
        loss = self.fc1.backward(loss, lr)
        return loss


class DataLoader:
    def __init__(self, x, y=None, batch_size=32) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.x):
            raise StopIteration
        if self.index+self.batch_size >= len(self.x):
            begin = self.index
            self.index = len(self.x)
            if self.y is None:
                return self.x[begin:]
            else:
                return self.x[begin:], self.y[begin:]
        else:
            begin = self.index
            self.index = begin+self.batch_size
            if self.y is None:
                return self.x[begin:self.index]
            else:
                return self.x[begin:self.index], self.y[begin:self.index]

    def __len__(self):
        return (len(self.y)+self.batch_size-1)//self.batch_size


loss_func = CrossEntropyLoss()

train_data = np.load("mnist_detail/train.npz")
train_x = train_data['x'].reshape((-1, 28*28))
y_target = train_data['y']
train_x_mean = train_x.mean()
train_x_std = train_x.std()
train_x = (train_x-train_x_mean)/(train_x_std+eps)
train_y = np.zeros((len(y_target), 10))
for i, j in enumerate(y_target):
    train_y[i][j] = 1.0

test_x = np.load("mnist_detail/test_x.npy").reshape((-1, 28 * 28))
test_x = (test_x-train_x_mean)/(train_x_std+eps)
test_y = np.load("mnist_detail/test_y.npy")

model = Model()

test_loader = DataLoader(test_x, test_y, 64)
train_loader = DataLoader(train_x, train_y, 64)


def test():
    count = 0
    for (x, y) in test_loader:
        out = model(x)
        pred = np.argmax(out, axis=1)
        count += (pred == y).sum()
    print(f"Test Acc {count/(len(test_y)):.4f}")


def train():
    count = 0
    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        out = model(x)
        loss = loss_func(out, y)
        model.backward(loss_func.backward(), 10)

        count += (np.argmax(out, axis=1) == np.argmax(y, axis=1)).sum()

    print(f"Train Acc {count/len(train_y):.4f}")  # f"train acc is :{k:.4f}" -> f+{} 中间可加括号插入


for i in range(40):
    train()
    test()
