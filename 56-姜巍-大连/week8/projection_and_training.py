import numpy as np


# 手撕课堂作业例子中的BP神经网络

# # 定义输入层,batch size = 1, 样本1的特征向量为x = [input_01, input_02]
# x = np.zeros(2)
# # 定义隐藏层神经元数量为2，与输入层全连接。权重关系矩阵(input_hide):
# # [[input_01->hide_z01,input_01->hide_z02], [input_02->hide_z01,input_02->hide_z02]] = [[w1,w2],[w3,w4]]
# input_to_hide = np.zeros((2, 2))
# bias_01 = 0.1
# hide_layer = np.zeros(2)
# # 定义输出层神经元数量为2，与隐藏层全连接。权重关系矩阵(hide_output):
# # [[hide_a01->hide_z01,input_a01->hide_z02], [input_a02->hide_z01,input_a02->hide_z02]] = [[w5,w6],[w7,w8]]
# hide_to_output = np.zeros((2, 2))
# bias_02 = 0.1
# output_layer = np.zeros(2)


# 定义层间计算方法(输入->加权求和->激活)激活函数选用sigmoid函数：sigmoid = lambda z: 1 / (1 - np.exp(-z))
def array_multiply(inp, w, bias):
    out_z = np.zeros(2)
    out_z[0] = inp[0] * w[0][0] + inp[1] * w[0][1] + bias
    out_z[1] = inp[0] * w[1][0] + inp[1] * w[1][1] + bias
    out_a = (lambda a, b: np.array([1 / (1 + np.exp(-out_z[a])), 1 / (1 + np.exp(-out_z[b]))]))(0, 1)
    return out_a


# # 损失函数选用均值平方差函数
# label为：y = np.zeros(2)
# MSE = 0.5 * ((y[0] - output_layer[0]) ** 2 + (y[1] - output_layer[1]) ** 2)

def evolution(inp, label, hide, out, i2h, h2o, ra=0.5):
    # 梯度下降法优化权重wi，需要求出MSE对各wi的一阶偏导数
    # 首先列出求导结果中几个重要常数:
    K_01 = (label[0] - out[0]) * out[0] * (1 - out[0])
    K_02 = (label[1] - out[1]) * out[1] * (1 - out[1])
    K_03 = hide[0]
    K_04 = 1 - hide[0]
    K_05 = hide[1]
    K_06 = 1 - hide[1]
    # 由此我们可以较简单地列出MSE对各wi的一阶偏导数
    m_wi = np.zeros(8)
    m_wi[0] = -(K_01 * h2o[0][0] + K_02 * h2o[1][0]) * K_03 * K_04 * inp[0]
    m_wi[1] = -(K_01 * h2o[0][0] + K_02 * h2o[1][0]) * K_03 * K_04 * inp[1]
    m_wi[2] = -(K_01 * h2o[0][1] + K_02 * h2o[1][1]) * K_05 * K_06 * inp[0]
    m_wi[3] = -(K_01 * h2o[0][1] + K_02 * h2o[1][1]) * K_05 * K_06 * inp[1]
    m_wi[4] = -K_01 * K_03
    m_wi[5] = -K_01 * K_05
    m_wi[6] = -K_02 * K_03
    m_wi[7] = -K_02 * K_05
    m = 0
    # 为输入层->隐藏层的[[w1,w2],[w3,w4]]更新权重
    for i in range(len(i2h)):
        for j in range(len(i2h)):
            i2h[i][j] -= ra * m_wi[m]
            m += 1
    # 为隐藏层->输出层的[[w5,w6],[w7,w8]]更新权重
    for p in range(len(h2o)):
        for q in range(len(h2o)):
            h2o[p][q] -= ra * m_wi[m]
            m += 1


if __name__ == '__main__':
    # 初始条件赋值
    x = np.array([0.05, 0.1])
    y = np.array([0.01, 0.99])
    input_to_hide = np.array([[0.15, 0.2], [0.25, 0.3]])
    hide_to_output = np.array([[0.4, 0.45], [0.5, 0.55]])
    bias_01, bias_02 = 0.35, 0.6
    # 进行第一次正向训练过程
    hide_layer = array_multiply(x, input_to_hide, bias_01)
    output_layer = array_multiply(hide_layer, hide_to_output, bias_02)
    MSE = 0.5 * ((y[0] - output_layer[0]) ** 2 + (y[1] - output_layer[1]) ** 2)
    t = 0
    # 迭代,假设要求对任意yi误差不超过1%，那么MSE <= 0.5 * (0.05 * 0.01) ** 2 = 0.000000125
    while MSE > 0.000000125:
        evolution(x, y, hide_layer, output_layer, input_to_hide, hide_to_output, ra=0.5)
        hide_layer = array_multiply(x, input_to_hide, bias_01)
        output_layer = array_multiply(hide_layer, hide_to_output, bias_02)
        MSE = 0.5 * ((y[0] - output_layer[0]) ** 2 + (y[1] - output_layer[1]) ** 2)
        t += 1

    print('输入层->隐藏层的各权重[[w1,w2],[w3,w4]]为：', input_to_hide)
    print('隐藏层值为：', hide_layer)
    print('隐藏层->输出层的各权重[[w5,w6],[w7,w8]]为：', hide_to_output)
    print('输出层值为：', output_layer)
    print('最终误差损失函数值为：', MSE)
    print('迭代次数为：', t)
