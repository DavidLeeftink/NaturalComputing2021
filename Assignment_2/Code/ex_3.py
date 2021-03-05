from sklearn import datasets
from kmeans import KMeans
from pso import PSO
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
data = np.array(iris['data'])

def quantization_error(centroids, data):
    n_clusters = centroids.shape[0]
    distances = np.sqrt(np.square(data - centroids[:, None]).sum(axis=2)).T
    clustering = np.argmin(distances,axis = 1)
    means = np.zeros((n_clusters,))
    n_non_empty_clusters = 0
    for i in range(n_clusters):
        if np.sum(clustering == i) > 0:
            n_non_empty_clusters+=1
            means[i] = np.mean(distances[clustering == i,i])
    return np.sum(means)/n_non_empty_clusters


# ## K means
# k=3
# params = {'k': k,
#           'data': np.array(iris['data']),
#           'centroids':np.random.randn(k,4)+np.mean(np.array(iris['data']), axis = 0)
#           }
# kmeans =  KMeans(params)
# for i in range(1000):
#     clust, cent = kmeans.step()
# print("KMeans Quantization error: ",quantization_error(cent, kmeans.data))


## Particles

def obj(X):
    return quantization_error(np.squeeze(X), data)
n_clusters = 100
X = np.full((n_clusters,3,4),np.mean(data,axis= 0))
X += np.random.randn(n_clusters,3,4)*0.2
V  = np.random.randn(n_clusters,3,4)*0.5
X_hat = X
g_hat = X[0]

params ={
    'n':n_clusters,
    'omega':0.7298,
    'a1':1.49618,
    'a2':1.49618,
    'X':X,
    'V':V,
    'X_hat':X_hat,
    'g_hat':g_hat,
    'objective_fn':obj,
    'maximize':False,
    'range':None
}
pso = PSO(params)
mins = np.min(data,axis = 0)
maxs = np.max(data, axis = 0)
print(mins.shape)
for h in range(200):
    pso.step()
    # fig,ax = plt.subplots(4,4, figsize= (15,15))
    # plt.tight_layout(pad=0.5)

    # for i in range(4):
    #     for j in range(4):
    #         if not i == j:
    #             ax[i,j].scatter(data[:,i],data[:,j],s=20, alpha = 0.5)
    #             ax[i,j].set_xlim(mins[i]-0.5,maxs[i]+0.5)
    #             ax[i,j].set_ylim(mins[j]-0.5,maxs[j]+0.5)
    #             for k in range(n_clusters):
    #                 ax[i,j].scatter(pso.X[k,:,i],pso.X[k,:,j],s=50)
                
    # for a in ax.flatten():
    #     a.set_xticks([])
    #     a.set_yticks([])
    # num =str(h).zfill(3)
    # plt.savefig(f"/home/guus/Uni/AI_Master/Years/1/sem2/NatCo/asgn/asgn2/swarmgif2/{num}.png")
    # # plt.show()
    # plt.close()

best = np.squeeze(pso.g_hat)
print(obj(best))
squared_distances = np.square(data - best[:, None]).sum(axis=2)
clusters = np.argmin(squared_distances,0)






