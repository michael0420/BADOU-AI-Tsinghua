# use sklearn  test pca


import  matplotlib.pyplot as plt   # 加载matplotlib 用于数据的可视化
from sklearn.decomposition import  PCA  # PCA #加载PCA 算法包
from sklearn.datasets import load_iris

data = load_iris()
y = data.target  # y 标签 0，1，2
x = data.data    # x 数据集 4 特征
pca = PCA(n_components=2) # 加载PCA 算法，设置降维后主成分数目为2
reduced_x =  pca.fit_transform(x) # 对样本进行降为后的数据
#print(y)
#print(x)
print(reduced_x) # 2 维特征
red_x,red_y = [],[]
blue_x,blue_y =[],[]
green_x,green_y=[],[]
# 通过循环开始分类
for i in range(len(reduced_x)):
   if  y[i] ==0:  # 标签为0
       red_x.append(reduced_x[i][0])
       red_y.append(reduced_x[i][1])
   elif y[i]==1:
       blue_x.append(reduced_x[i][0])
       blue_y.append(reduced_x[i][1])
   else:
       green_x.append(reduced_x[i][0])
       green_y.append(reduced_x[i][1])
#可视化
plt.scatter(red_x,red_y,c="r",marker="x")
plt.scatter(blue_x,red_y,c="b",marker="o")
plt.scatter(green_x,green_y,c="g",marker=".")
plt.show()

