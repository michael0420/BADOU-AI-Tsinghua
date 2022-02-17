import random
import numpy as np

"""定义一个距离计算函数"""
def calculate_distance(node1,node2):
    distance=(node2[0]-node1[0])**2+(node2[1]-node1[1])**2
    distance=distance**0.5   #开平方用**
    return distance

"""定义一个函数，初次得到分类后的簇"""
def first_cycle(src_data,k,first_point):
    new_set=dict()   #dict()为定义一个字典（dict）,用于存放分类后的簇
    for a in range(k):   #该for循环是为了将字典里的每一个元素定义为列表[]格式
        new_set[a] = []

        #对源数据中的每一个点，初始分类到对应的簇中
    for i in range(len(src_data)):
        current_node=src_data[i]
        distance1=calculate_distance(first_point[0],current_node)

        #该循环是将源数据中的第i个点，分别计算与定义好的中心点的距离，分别比较，得到最小距离并存入对应的集合中
        for n in range(k):   #
            center_point=first_point[n]
            distance=calculate_distance(current_node,center_point)
            if distance<=distance1:
                distance1=distance
                key=n
        new_set[key].append(src_data[i])
    return new_set

"""定义一个新的质心求取函数"""
def calculate_center(new_set,k):
    # 定义一个字典，dict.fromkeys(range(k),[])用于对字典内的每一个元素赋予初始量，range(k)代表键值，[]代表初始值
    center_p=dict.fromkeys(range(k),[])
    for i in range(k):
        center_p[i]=np.mean(new_set[i],axis=0)  #np.mean()为平均值求取函数
    return center_p

"""定义一个函数，以新的质心为中心进行簇的分类"""
def set_cycle(src_data,k,center):
    #定义一个字典，每个键值赋予[]的值。{键_表达式:值_表达式 for 表达式 in 可迭代对象}
    c_new_set={n:[] for n in range(k)}
    # c_new_set=dict()
    # for a in range(k):
    #     c_new_set[a] = []
    for i in range(len(src_data)):
        current_node=src_data[i]
        distance1=calculate_distance(center[0],current_node)

        for n in range(k):
            center_point=center[n]
            distance=calculate_distance(current_node,center_point)
            if distance<=distance1:
                distance1=distance
                key=n
        c_new_set[key].append(src_data[i])
    return c_new_set

"""定义一个函数，从源数据中随机取得初始中心点"""
def get_first_point(src_data,k):
    first_point=list(src_data)
    return random.sample(first_point,k)   #random.sample(seq,k)函数用于从定义序列seq中，抓取出定义长度k的元素

"""定义k-Means函数"""
def k_Means_diy(src_data,k):
    first_point=get_first_point(src_data,k)
    new_set = first_cycle(src_data, k, first_point)
    # 定义一个字典，dict.fromkeys(range(k),[])用于对字典内的每一个元素赋予初始量，range(k)代表键值，[]代表初始值
    temp_set = dict.fromkeys(range(k),[])
    #用while函数进行条件判断，当前后两次簇分类获得的集合相同时，即聚类完成
    while new_set != temp_set:
        center_p = calculate_center(new_set, k)
        temp_set = new_set
        new_set = set_cycle(src_data, k, center_p)
    return new_set

"""随即定义数据测试"""
k=3
src_data=[[1,1], [1,2],[2,1],[3,3],[26,28],[4,3],[6,6],[6,7],[1,3],[2,2]]
test=k_Means_diy(src_data,k)
print(test)