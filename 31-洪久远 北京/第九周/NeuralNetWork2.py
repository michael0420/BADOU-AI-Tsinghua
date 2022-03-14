import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lr = learningRate
        # 以上为基本参数初始化
        self.wih = (np.random.normal(loc=0.0, scale=pow(self.hNodes, -0.5), size=(self.hNodes, self.iNodes)))
        self.who = (np.random.normal(loc=0.0, scale=pow(self.oNodes, -0.5), size=(self.oNodes, self.hNodes)))
        # 以上为权重矩阵初始化，wih为输入-隐藏层权重矩阵，who为隐藏-输出权重矩阵
        # loc为均值，scale为标准差
        self.activationFunction = lambda x: scipy.special.expit(x)
        # 以上为激活函数sigmoid

    def train(self, inputList, targetList):
        inputs = np.array(inputList, ndmin=2).T
        targets = np.array(targetList, ndmin=2).T
        hiddenInputs = np.dot(self.wih, inputs)
        # 以上为隐藏层的输入
        hiddenOutputs = self.activationFunction(hiddenInputs)
        #  以上为隐藏层经过激活后的输出
        finalInputs = np.dot(self.who, hiddenOutputs)
        # 以上为输出层的输入
        finalOutputs = self.activationFunction(finalInputs)
        # 以上为最终输出
        outputErrors = targets - finalOutputs
        hiddenErrors = np.dot(self.who.T, outputErrors * finalOutputs * (1 - finalOutputs))
        self.who += self.lr * np.dot((outputErrors * finalOutputs * (1 - finalOutputs)), np.transpose(hiddenOutputs))
        self.wih += self.lr * np.dot((hiddenErrors*hiddenOutputs*(1-hiddenOutputs)),np.transpose(inputs))


    def query(self,inputs):
        hiddenInputs=np.dot(self.wih,inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)
        finalInputs = np.dot(self.who, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)
        print(finalOutputs)
        return finalOutputs


inputNodes=784
hiddenNodes=200
outputNodes=10
learningRate=0.1
n=NeuralNetWork(inputNodes,hiddenNodes,outputNodes,learningRate)

#以下读取训练数据
trainDataFile=open("dataset/mnist_train.csv",'r')
trainDataList=trainDataFile.readlines()
trainDataFile.close()

#加入epochs，即循环次数

epochs=5
for e in range(epochs):
    for record in trainDataList:
        allValues=record.split(',')
        inputs=(np.asfarray(allValues[1:]))/255.0*0.99+0.01
        #此为图片与数值的对应关系
        targets=np.zeros(outputNodes)+0.01
        targets[int(allValues[0])]=0.99
        n.train(inputs,targets)

testDataFile=open("dataset/mnist_test.csv")
testDataList=testDataFile.readlines()
testDataFile.close()

scores=[]
for record in testDataList:
    allValuesTest=record.split(',')
    correctNumber=int(allValuesTest[0])
    print("该图片对应数字为：",correctNumber)
    inputs=(np.asfarray(allValuesTest[1:]))/255.0*0.99+0.01
    outputs=n.query(inputs)
    label=np.argmax(outputs)
    print("网络认为的图片为：",label)
    if label==correctNumber:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

scoresArray=np.asarray(scores)
print('performance=',scoresArray.sum()/scoresArray.size)
