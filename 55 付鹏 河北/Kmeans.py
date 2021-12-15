from sklearn.cluster import KMeans
X = [[1, 3],
     [4, 5],
     [6, 8],
     [9, 6],
     ]
print(X)
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)
print(clf)
print("y_pred = ", y_pred)
import matplotlib.pyplot as plt
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)
plt.scatter(x, y, c=y_pred, marker='x')
plt.scatter(x, y, c=y_pred, marker='o')
plt.scatter(x, y, c=y_pred, marker='*')
plt.title("Kmeans-Basketball Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend(["A", "B", "C"])
plt.show()

