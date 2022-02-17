import numpy as np
#从数据集中随机选取k个样本
dataSet = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]
print(len(dataSet))
#print(dataSet[8][1])
A = np.array(dataSet)
k = 3
B = np.random.choice([x for x in range(len(dataSet))], k, replace = False)
C= []
for i in range(k):
    C.append(A[B[i]])
C = np.array(C)#C是存放样本集中随机选取得K个样本组成的数组
print(C,C[0,1],C[2][0])
B1, B2, B3 = [], [], []# 存放样本集各样本点距离选取质心的距离列表
n = 15
while n > 0:
    B1, B2, B3 = [], [], []# 存放距离质心较近的点的列表
    B11, B22, B33 = [], [], []#存放质心聚类的样本
    for i in range(len(dataSet)):
        B1.append(np.sqrt((dataSet[i][0] - C[0][0])**2 + (dataSet[i][1] - C[0][1])**2))
        B2.append(np.sqrt((dataSet[i][0] - C[1][0])**2 + (dataSet[i][1] - C[1][1])**2))
        B3.append(np.sqrt((dataSet[i][0] - C[2][0])**2 + (dataSet[i][1] - C[2][1])**2))
        if (B1[i] < B2[i]) and (B1[i] < B3[i]):
            B11.append(dataSet[i])
        elif (B2[i] < B1[i]) and (B2[i] < B3[i]):
            B22.append(dataSet[i])
        elif (B3[i] < B1[i]) and (B3[i] < B2[i]):
            B33.append(dataSet[i])
    n = n-1
    print(B11, B22, B33)
    C = [np.average(np.array(B11), axis=0), np.average(np.array(B22), axis=0), np.average(np.array(B33), axis=0)]
    print(C)

#%%y_pred 存放样本集中的数据属于哪一类的索引
y_pred = [0 for x in range(0,len(dataSet))]
for i in range(len(dataSet)):
    if dataSet[i] in B11:
        y_pred[i] = 0
    elif dataSet[i] in B22:
        y_pred[i]= 1
    else:
        y_pred[i] = 2
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt

#获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in dataSet]
print (x)
y = [n[1] for n in dataSet]
print (y)

'''
绘制散点图
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=y_pred, marker='x')

#绘制标题
plt.title("Kmeans-Basketball Data")

#绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

#设置右上角图例
plt.legend(["A","B","C"])

#显示图形
plt.show()
