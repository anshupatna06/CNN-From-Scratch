import numpy as np

def initialize_filters(F, C, FH, FW):
    return np.random.randn(F, C, FH, FW) * 0.01

def initialize_bias(F):
    return np.zeros(F)

def print_shape(name, x):
    print(f"{name} shape = {x.shape}")

