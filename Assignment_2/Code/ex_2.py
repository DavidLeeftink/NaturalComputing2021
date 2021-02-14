import numpy as np
import matplotlib.pyplot as plt
from pso import PSO


n_particles = 1
X = np.array([[20]])
params = {
    'n':n_particles,
    'omega':0.8,
    'a1':1,
    'a2':1,
    'X':X,
    'V':np.array([[10]]),
    'X_hat':X,
    'g_hat':X,
    'objective_fn':np.square,
    'maximize':True,
    'range':None
}
pso = PSO(params)
n_steps = 100
steps = np.zeros(n_steps+1)
for i in range(n_steps):
    pso.step()
    steps[i+1] = pso.X
plt.plot(steps)
plt.show()