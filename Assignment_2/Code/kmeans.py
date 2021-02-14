import numpy as np

class KMeans:
    def __init__(self, params):
        self.k = params['k']
        self.data = params['data']
        self.centroids = params['centroids']
        self.N,self.M = self.data.shape
    def step(self):
        squared_distances = np.square(self.data - self.centroids[:, None]).sum(axis=2)
        clusters = np.argmin(squared_distances,0)
        new_centroids = self.centroids.copy()
        for i in range(self.k):
            if np.sum(clusters == i) > 0:
                new_centroids[i] = np.mean(self.data[clusters == i])
        returncentroids = np.copy(self.centroids)   
        self.centroids = new_centroids
        return clusters, returncentroids
