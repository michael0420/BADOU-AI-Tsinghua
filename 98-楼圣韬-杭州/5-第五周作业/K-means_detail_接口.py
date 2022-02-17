import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = 'KaiTi'


def isin(lis, k):
    for i in lis:
        if k == i:
            return False
    return True


def makepoint(n, k):  # 随机生成n个不重复点，交叉结构
    sum, lis, ri = 0, [], 0
    while sum != n / 2:
        if sum % (n / k) == 0:
            ri = random.randint(1, 900)
        x = random.uniform(ri, ri + 200)
        y = random.uniform(ri, ri + 200)
        if isin(lis, [x, y]):
            lis.append([x, y])
            sum += 1
    while sum != n:
        if sum % (n / k) == 0:
            ri = random.randint(1, 900)
        x = random.uniform(ri, ri + 200)
        y = random.uniform(1000 - ri, 1000 - (ri + 200))
        if isin(lis, [x, y]):
            lis.append([x, y])
            sum += 1
    return lis


def printpoint(a, o):  # 打印散点，并将散点返回
    xz, yz = [], []
    for i in range(len(a)):
        xz.append(a[i][0])
        yz.append(a[i][1])
    h = ['b', 'c', 'k', 'g', 'm', 'y']
    plt.scatter(xz, yz, marker='*', c=h[o])
    plt.title("随机生成散点图")
    return a


def avepoint(a):
    x, y = 0, 0
    for i in a:
        x += i[0]
        y += i[1]
    return [x / len(a), y / len(a)]


def dis(n, b):
    return pow((b[0] - n[0]) ** 2 + (b[1] - n[1]) ** 2, 1 / 2)


def K_means(lis, k):  # 传入散点列表lis，并且分成k组
    tot = []
    for i in range(k):  # 随机生成k个数据中心
        tot.append([])
        tot[i].append(lis[random.randint(0, len(lis) - 1)])
    flag = 1
    t = []
    for i in range(k):  # 初始化数据中心
        t.append(avepoint(tot[i]))

    for i in lis:  # 初始化cluster
        min, mi = 10000, 0
        for j in range(k):
            if dis(t[j], i) < min:
                mi = j
        if isin(tot[mi], i):
            tot[mi].append(i)
    x, y, flag = 0, 0, 1
    while 1:
        p = 0
        for i in range(k):  # 更新数据中心
            if t[i] == avepoint(tot[i]):
                p += 1
            t[i] = avepoint(tot[i])
        if p == k:
            break
        for i in range(k):
            for j in tot[i]:
                p1 = i
                h = dis(j, t[i])
                for p in range(k):
                    if dis(j, t[p]) < h:
                        h = dis(j, t[p])
                        p1 = p
                if p1 != i:
                    tot[i].remove(j)
                    tot[p1].append(j)
        print()
    return tot


a = printpoint(makepoint(2000, 2000), 1)
plt.show()
tot = K_means(a, 5)
for i in range(5):
    printpoint(tot[i], i)
plt.show()

print("---------------------# 接口 #---------------------")
"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""
from sklearn.cluster import KMeans

clf = KMeans(n_clusters=5)   # kmeans 聚类
y_pred = clf.fit_predict(a)

print(clf)
# 输出聚类预测结果
print("y_pred = ", y_pred)

# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in a]
y = [n[1] for n in a]
plt.scatter(x, y, c=y_pred, marker='x')
plt.title("Kmeans-Basketball Data")
plt.legend(["A", "B", "C","D","E"])
plt.show()
