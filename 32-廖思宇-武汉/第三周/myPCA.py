import numpy as np


class myPCA():
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centreX = []
        self.T = []

    def fit_transform(self):
        self.centreX = self.centralized()
        featureValue, featureVector = np.linalg.eig(np.dot(self.centreX.T, self.centreX)/self.centreX.shape[0])
        ind = np.argsort(-featureValue)[:self.K]
        self.T = featureVector[:, ind]
        print("贡献率：", featureValue[ind][0]/np.sum(featureValue), featureValue[ind][1]/np.sum(featureValue))
        print("转换矩阵 \n", self.T)
        return self.centreX.dot(self.T)

    def centralized(self):
        centrX = self.X - np.array([np.mean(col) for col in self.X.T])
        return centrX


if __name__ == "__main__":
    # X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    pca = myPCA(X, 2)
    newX = pca.fit_transform()
    print(newX)
