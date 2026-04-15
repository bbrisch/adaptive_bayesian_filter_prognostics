import numpy as np
from scipy.stats import norm


class SIR_PF():
    def __init__(self, state_fn, N_particles, obs_params):
        
        self.state_fn = state_fn
        self.N_particles = N_particles
        self.obs_params = obs_params
        
        
    def init_state(self, x_init):
        self.x = x_init
        self.w = np.ones_like(x_init)/self.N_particles
        
    def _gaussian_likelihood(self, y, x_next):
        return norm.pdf(y - x_next,*self.obs_params )
        
    def mean(self):
        return (self.x*self.w).sum()
    
    def quantile(self, q):
        return np.quantile(self.x, q, weights=self.w, method='inverted_cdf')
        
        
    def update(self,y):
        # 1. Propagate particles with state equation 
        x_next = self.state_fn(self.x)
        
        # 2. Compute the new wheights (assuming p(x_t|x_1:t-1) = q(x_t|x_1:t-1))
        w_next = self.w*self._gaussian_likelihood(y,x_next)
        w_next = w_next/w_next.sum()
        
        # 3. Resampling to avoid degenerancy
        N_eff = 1/((w_next**2).sum())
        if self.N_particles <= N_eff:
            print('Resampling')
            x_next = np.random.choice(x_next, self.N_particles, replace=True, p=w_next)
            w_next = np.ones_like(w_next)/self.N_particles
        
        # 4. Asign the new state
        self.x = x_next
        self.w = w_next
        
        
        
        
