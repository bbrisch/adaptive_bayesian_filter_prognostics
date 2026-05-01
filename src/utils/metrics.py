import numpy as np


def expected_rul(x,w):
    return (x*w).sum()

def sample_crps(y,x,w):
    term_1 = (abs(x-y)*w).sum()
    
    subset_1 = np.random.choice(x, len(x)-1, p=w)
    subset_2 = np.random.choice(x, len(x)-1, p=w)
    term_2 = (abs(subset_1-subset_2)).mean()
    
    return term_1 - term_2/2