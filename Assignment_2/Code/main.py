import numpy as np
import matplotlib.pyplot as plt

from pso import PSO
def target(X):
    return np.sum(-X*np.sin(np.sqrt(np.abs(X))),1)

# xx = yy = np.linspace(-500,500,50)
# X,Y = np.meshgrid(xx,yy)
# Z = target(np.stack((X.ravel(),Y.ravel()),1))
# Z = np.reshape(Z,(50,50))

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X,Y,Z)
# plt.show()



n_particles = 100
X = np.random.uniform(-500,500,(n_particles,2))
params = {
    'n':n_particles,
    'omega':0.5,
    'a1':0.1,
    'a2':0.1,
    'X':X,
    'V':np.random.uniform(-1,1,(n_particles,2)),
    'X_hat':X,
    'g_hat':np.array([[0,0]]),
    'objective_fn':target,
    'maximize':True,
    'range':((-500,500),(-500,500))
}

pso = PSO(params)
for i in range(100):
    pso.step()
    if i%10 ==0:
        plt.scatter(pso.X[:,0],pso.X[:,1])
        plt.xlim((-500,500))
        plt.ylim((-500,500))
        plt.show()