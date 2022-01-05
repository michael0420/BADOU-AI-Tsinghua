import numpy as np
from sklearn.cluster import KMeans
import sklearn.datasets 
import matplotlib.pyplot as plt

X, y = sklearn.datasets.make_moons(500, noise=0.1,random_state=1)


# plt.scatter(X[:,0],X[:,1])
# plt.show()
# pass

##这一块是sklearn的包实现kmeans
# clf=KMeans(n_clusters=2)
# clf.fit(X)

# centers=clf.cluster_centers_
# labels=clf.labels_

# for i in range(len(labels)):
#     plt.scatter(X[i][0],X[i][1],c=('g' if labels[i]==0 else 'r'))
# plt.scatter(centers[0][0],centers[0][1],c='g',marker='*')
# plt.scatter(centers[1][0],centers[1][1],c='r',marker='>')
# plt.show()
# pass

class kmeans:
    def __init__(self,k) -> None:
        self.k=k
    def dis(self,p1,p2):
        assert len(p1)==len(p2)
        return np.sqrt(np.sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))
    def fit(self,data:np.ndarray,max_iter=500,tol=1e-4):
        centers=list(data[np.random.choice(len(data),self.k)])
        iter_num=0
        while iter_num<=max_iter:
            labels=[]
            if iter_num>0:
                centers=new_centers[:]
            for i in range(len(data)):
                dis_list=[self.dis(data[i],_) for _ in centers]
                label=dis_list.index(min(dis_list))
                labels.append(label)
            new_centers=[]
            for lb in set(labels):
                lb_data=data[np.array(labels)==lb]
                center_lb=np.mean(lb_data,axis=0) 
                new_centers.append(center_lb)
            iter_num+=1
            sort_center=sorted(centers,key=lambda x:x[0])
            sort_n_center=sorted(new_centers,key=lambda x:x[0])
            if np.all([np.all(i==j) for i,j in zip(sort_center,sort_n_center)]):
                break
        self.visual(data,labels,new_centers)
        return new_centers,labels
    def visual(self,data,labels,centers):
        for i in range(len(labels)):
            plt.scatter(data[i][0],data[i][1],c=('g' if labels[i]==0 else 'r'))
        plt.scatter(centers[0][0],centers[0][1],c='g',marker='*')
        plt.scatter(centers[1][0],centers[1][1],c='r',marker='>')
        plt.show()

km=kmeans(k=2)
km.fit(X)