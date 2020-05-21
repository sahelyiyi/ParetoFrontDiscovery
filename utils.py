import numpy as np


def get_uniform_random_unit_vector(d):
    v = np.random.rand(d)
    v_hat = v / np.linalg.norm(v)
    return v_hat


def get_uniform_positive_random_unit_vector(d):
    v = np.random.rand(d)
    while len(v[v==0]):
        v = np.random.rand(d)
    v_hat = v / np.linalg.norm(v)
    return v_hat
