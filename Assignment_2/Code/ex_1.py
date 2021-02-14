import numpy as np
import matplotlib.pyplot as plt
from pso import PSO
def target(X):
    return np.sum(-X*np.sin(np.sqrt(np.abs(X))),1)
    
# Note: change the PSO to have a constant r1 and r2
def get_pso_with_omega(omega):
    n_particles = 3
    X = np.array([[-400,-400],[-410,-410],[-415,-415]])
    params = {
        'n':n_particles,
        'omega':omega,
        'a1':1,
        'a2':1,
        'X':X,
        'V':np.full((3,2),-50),
        'X_hat':X,
        'g_hat':X[None,2],
        'objective_fn':target,
        'maximize':True,
        'range':((-500,500),(-500,500))
    }
    return PSO(params)

pso = get_pso_with_omega(2)
fitness_0 = target(pso.X)

for omega in [2,0.5,0.1]:
    pso = get_pso_with_omega(omega)
    pso.step()
    print(target(pso.X).round(2))
    print(pso.X.round(2))
