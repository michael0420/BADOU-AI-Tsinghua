from sklearn.datasets.base import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


X, Y = load_iris(return_X_y=True)
newX = PCA(n_components = 2).fit_transform(X)
class1_X, class2_X, class3_X = [], [], []
class1_Y, class2_Y, class3_Y = [], [], []
for i in range(len(Y)):
    if Y[i] == 0:
        class1_X.append(newX[i][0])
        class1_Y.append(newX[i][1])
    elif Y[i] == 1:
        class2_X.append(newX[i][0])
        class2_Y.append(newX[i][1])
    else:
        class3_X.append(newX[i][0])
        class3_Y.append(newX[i][1])
plt.scatter(class1_X, class1_Y, c='r', marker='s')
plt.scatter(class2_X, class2_Y, c='g', marker='p')
plt.scatter(class3_X, class3_Y, c='b', marker='x')
plt.show()


