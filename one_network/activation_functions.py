import numpy as np

def relu(X):
    # return np.maximum(0, X)
    return np.multiply(X,(X>0))

def relu_deriv(X):
    return (X > 0) * 1