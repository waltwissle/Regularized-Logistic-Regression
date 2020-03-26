# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:16:45 2020

@author: Walter Aburime
"""

import numpy as np
import sigmoid

def gradient(theta, X, Y, l):
    
    m,n = X.shape
    theta = theta.reshape((n,1))
    grad = np.zeros((theta.shape))
    Y = Y.reshape((m,1))
    #grad = np.zeros((theta.shape))
    h_theta = sigmoid.sigmoid(X @ theta) #the hypothesis h(theta) = 1/(1 + e**(z))
    
    grad[0,:] = (1/m) * (h_theta - Y).T @ X[:,0]
    
    grad[1:,:] = (((1/m)* (h_theta - Y).T @ X[:,1:]) + ((l/m)* theta[1:,:]).T).T
    return grad
    