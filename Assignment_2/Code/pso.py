import numpy as np


class PSO:
    def __init__(self, params):
        self.n = params['n']
        self.omega = params['omega']
        self.a1 = params['a1']
        self.a2 = params['a2']
        self.X = params['X']
        self.V = params['V']
        self.X_hat = params['X_hat']
        self.g_hat = params['g_hat']
        self.objective_fn = params['objective_fn']
        self.maximize = params['maximize']
        self.flip = 1 if self.maximize else -1
        if params['range']:
            self.clip = True
            self.x_lim = params['range'][0]
            self.y_lim = params['range'][0]
        else:
            self.clip = False
]

    def step(self):
        for i in range(self.n):
            r1, r2 = np.random.rand(2)
            self.V[i] = self.omega*self.V[i]+self.a1*r1 * \
                (self.X_hat[i] - self.X[i]) + self.a2*r2*(self.g_hat-self.X[i])
            self.X[i] = self.X[i] + self.V[i]
        
        if self.clip:
            self.X[:,0] = np.clip(self.X[:,0],*self.x_lim)
            self.X[:,1] = np.clip(self.X[:,1],*self.y_lim)

        for i in range(self.n):
            cur_obj = self.flip*self.objective_fn(self.X[None,i])
            if cur_obj > self.flip*self.objective_fn(self.X_hat[None,i]):
                self.X_hat[i] = self.X[i]
            if cur_obj > self.flip*self.objective_fn(self.g_hat):
                self.g_hat = self.X[None,i]
