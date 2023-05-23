import numpy as np

"""
This function computes the output for the activation chosen
"""

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

def softmax(z):
    z=z-np.max(z)
    num=np.exp(z)
    den=np.sum(num,axis=0)
    return num/den

def relu(z):
    return (z>0)*(z) + 0.01*((z<0)*z)


