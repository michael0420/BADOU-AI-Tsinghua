from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

X = [[1,2], [3,2], [4,4], [1,2], [1,3]] # 原始数据
'''
linkage(y, method='single', metric='euclidean', optimal_ordering=False)函数参数如下：
    返回值：层次聚类编码为一个linkage矩阵，这个矩阵由四列组成:
            第一和第二列:分别为聚类簇的编号，在初始状态下每个初始值编号0~n-1，以后每生成一个新的聚类簇，就在此基础上增加一对
                      新的聚类簇进行标识。
            第三列标识两个聚类簇的距离.
            第四列表示新生成的聚类簇的元素个数 .
    
    参数：
    y：  可以是1维的压缩向量(距离向量)，也可以是2维观测向量(坐标矩阵)，如果y为1维压缩向量，则y必须是n个初始观测值的组合，
        n是坐标矩阵中成对的观测值
    method: 计算簇之间距离的方法，可选择：
            single:最邻近点算法，比较两个簇中所有点之间的距离，然后取最小距离作为簇之间的距离。
            complete:和single一样，比较两个簇中所有点之间的距离，但是这个取最大距离作为簇之间的距离。
            average:叫做UPGMA法(非加权组平均法)
            weighted:叫做WPGMA(加权分组平均法)
            centroid:叫做UPGMC算法
            median：当两个簇组成一个新簇的时候，原来的两个簇的质心均值为新簇的质心，WPGMC算法。
            ward: 沃德方差最小化算法              
'''
Z = linkage(X, 'ward')
print(Z)
'''
fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)函数参数：
    返回值：得到的平面聚类，也就是层次聚类结果。为一个数组，长度为原始数据长度，每个数组元素对应相应原始数据所处的簇
    
    参数：
    Z：linkage函数得到的矩阵，包含了层次聚类的各层信息
    t：层次聚类的阈值，也就是形成平面簇的阈值。
    criterion：形成平面簇的标准，可以为：
                inconsistent：
                distance：
                maxclust：
                monocrit：
                maxclust_monocrit：
    

'''
f = fcluster(Z, 4, 'distance')
print(f)
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)

plt.show()






