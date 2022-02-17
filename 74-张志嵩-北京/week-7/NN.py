# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:17:30 2022

@author: Administrator
"""
import numpy as np
import scipy.special


class NN():
    def __init__(self, input_node, hidden_node, class_num, lr):
        self.input_node = input_node
        self.output_node = class_num
        self.hidden_node = hidden_node
        self.lr = lr
        self.sigmoid = lambda x: scipy.special.expit(x)

        self.wih = np.random.normal(0.0, pow(self.hidden_node, -0.5), (self.hidden_node, self.input_node))
        self.who = np.random.normal(0.0, pow(self.output_node, -0.5), (self.output_node, self.hidden_node))

    def query(self, x):
        hidden_input = np.dot(self.wih, x)
        hidden_output = self.sigmoid(hidden_input)
        final_output = np.dot(self.who, hidden_output)
        final_output = self.sigmoid(final_output)
        return final_output

    def train(self, x, targets, batch, rate):
        x = x.T  # reshape((-1,batch))
        targets = targets.T  # reshape((-1,batch))

        hidden_input = np.dot(self.wih, x)
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(self.who, hidden_output)
        final_output = self.sigmoid(final_input)

        outputs_errors = pow(targets - final_output, 2)
        # print('loss is: ',outputs_errors.sum())
        hidden_derivate = np.dot(self.who.T, -0.5 * (targets - final_output) * (final_output) * (1 - final_output))

        self.who -= self.lr * np.dot(-0.5 * (targets - final_output) * (final_output) * (1 - final_output),
                                     hidden_output.T)
        self.wih -= self.lr * np.dot(hidden_derivate * hidden_output * (1 - hidden_output), x.T)
        return outputs_errors.sum()


def main():
    input_node = 784
    hidden_node = 200
    class_num = 10
    lr = 0.01
    batch = 16
    model = NN(input_node, hidden_node, class_num, lr)

    training_data_file = open("/home/uers/desk_B/八斗/dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 加入epocs,设定网络的训练循环次数
    epochs = 300
    for e in range(epochs):
        loss, total = 0.0, 0
        inputs, targets = [], []
        for i, train_data in enumerate(training_data_list):
            data = train_data.split(',')
            inputs_t = (np.array(data[1:]).astype(float)) / 255.0 * 0.99 + 0.01
            # 设置图片与数值的对应关系
            targets_t = np.zeros(class_num) + 0.01
            targets_t[int(data[0])] = 0.99
            inputs += list(inputs_t)
            targets += list(targets_t)
            # np.hstack((inputs,inputs_t))
            # np.hstack((targets,targets_t))
            if (i + 1) % batch == 0:
                inputs = np.array(inputs).reshape((batch, -1))
                targets = np.array(targets).reshape((batch, -1))
                loss += model.train(inputs, targets, batch, 1 - (e + 1) / epochs)
                total += 1
                print('epoch %d/%d  loss is: %f  lr is: %f' % (
                e + 1, epochs, loss / (total * batch), lr * (1 - (e + 1) / epochs)))
                inputs, targets = [], []

    test_data_file = open("/home/uers/desk_B/八斗/dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for test_data in test_data_list:
        data = test_data.split(',')
        target = int(data[0])
        print("该图片对应的数字为:", target)
        inputs = np.array(data[1:]).astype(float) / 255.0 * 0.99 + 0.01
        outputs = model.query(inputs)
        pre = np.argmax(outputs)
        print("网络认为图片的数字是：", pre)
        if pre == target:
            scores.append(1)
        else:
            scores.append(0)
    print('scores is: ', scores)
    accuracy = np.sum(scores) / len(scores)
    print("prediction accuracy is: ", accuracy)


if __name__ == "__main__":
    main()