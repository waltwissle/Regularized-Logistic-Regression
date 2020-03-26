"""
Created on Sun Mar 22 13:27:16 2020

@author: Walter Aburime
"""

import numpy as np


def mapFeature(x1, x2):
    degree = 6
    x1 = x1.reshape(len(x1),1)
    x2 = x2.reshape(len(x1),1)
    out = np.ones((x1.shape[0],1))
    
    
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            new = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, new, axis = 1)
    return out