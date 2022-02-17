import cv2
import numpy as np
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self):
        self.img = cv2.imread('/home/uers/desk_B/八斗/week2/lenna.png')
        self.shape = self.img.shape[:2]

    def gray(self):
        img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        img_g = img.reshape((self.shape[0]*self.shape[1], 1))
        img_g = np.float32(img_g)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(img_g, 4, None, criteria, 10, flags)
        dst = labels.reshape((self.shape[0], self.shape[1]))
        titles = [u'src', u'dst']
        images = [img, dst]
        for i in range(2):
            plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def athlete(self,X):
        from sklearn.cluster import KMeans
        clf = KMeans(n_clusters=3)
        y_pred = clf.fit_predict(X)
        print(clf)
        print("y_pred = ", y_pred)
        x = [n[0] for n in X]
        print(x)
        y = [n[1] for n in X]
        print(y)
        plt.scatter(x, y, c=y_pred, marker='x')

        # 绘制标题
        plt.title("Kmeans-Basketball Data")
        plt.xlabel("assists_per_minute")
        plt.ylabel("points_per_minute")
        plt.legend(["A", "B", "C"])
        plt.show()

    def rgb(self):
        data = self.img.reshape((-1,3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
        compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
        compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
        compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
        compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)
        centers2 = np.uint8(centers2)
        res = centers2[labels2.flatten()]
        dst2 = res.reshape((self.img.shape))

        centers4 = np.uint8(centers4)
        res = centers4[labels4.flatten()]
        dst4 = res.reshape((self.img.shape))

        centers8 = np.uint8(centers8)
        res = centers8[labels8.flatten()]
        dst8 = res.reshape((self.img.shape))

        centers16 = np.uint8(centers16)
        res = centers16[labels16.flatten()]
        dst16 = res.reshape((self.img.shape))

        centers64 = np.uint8(centers64)
        res = centers64[labels64.flatten()]
        dst64 = res.reshape((self.img.shape))
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
        dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
        dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
        dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
        dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

        titles = [u'src', u'dst K=2', u'dst K=4',
                  u'dst K=8', u'dst K=16', u'dst K=64']
        images = [img, dst2, dst4, dst8, dst16, dst64]
        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == "__main__":
    kmeans = KMeans()
    #kmeans.gray()
    # X = [[0.0888, 0.5885],
    #      [0.1399, 0.8291],
    #      [0.0747, 0.4974],
    #      [0.0983, 0.5772],
    #      [0.1276, 0.5703],
    #      [0.1671, 0.5835],
    #      [0.1306, 0.5276],
    #      [0.1061, 0.5523],
    #      [0.2446, 0.4007],
    #      [0.1670, 0.4770],
    #      [0.2485, 0.4313],
    #      [0.1227, 0.4909],
    #      [0.1240, 0.5668],
    #      [0.1461, 0.5113],
    #      [0.2315, 0.3788],
    #      [0.0494, 0.5590],
    #      [0.1107, 0.4799],
    #      [0.1121, 0.5735],
    #      [0.1007, 0.6318],
    #      [0.2567, 0.4326],
    #      [0.1956, 0.4280]
    #      ]
    # kmeans.athlete(X)
    kmeans.rgb()