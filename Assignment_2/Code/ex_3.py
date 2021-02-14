from sklearn import datasets
from kmeans import KMeans
import matplotlib.pyplot as plt
import numpy as np
iris = datasets.load_iris()


def quantization_error(centroids, data):
    n_clusters = centroids.shape[0]
    distances = np.sqrt(np.square(data - centroids[:, None]).sum(axis=2))
    clustering = np.argmin(distances, 0)
    sums = np.zeros((n_clusters,))
    for i in range(n_clusters):
        cluster_kard = np.sum(clustering == i)
        if cluster_kard > 0:
            sums[i] = np.sum(distances[clustering == i])/cluster_kard
    return np.sum(sums)/n_clusters


## K means
k=3
params = {'k': k,
          'data': np.array(iris['data']),
          'centroids':np.random.randn(k,4)
          }

kmeans =  KMeans(params)
for i in range(100):
    clust, cent = kmeans.step()

fig,ax = plt.subplots(4,4, figsize =(15,15))
for i in range(4):
    for j in range(4):
        if not i == j:
            ax[i,j].scatter(kmeans.data[:,i],kmeans.data[:,j], c=clust, s = 2)
plt.show()

print("KMeans Quantization error: ",quantization_error(kmeans.centroids, kmeans.data))