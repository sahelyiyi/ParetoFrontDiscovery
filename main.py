import random
import math
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

from ZDT1 import f_func, grad_f_func, grad2_f_func
from constants import N_S, N_i, LAMBDA_P, LAMBDA_S, D, d, K
from homotopy.algorithm import run_homotopy
from utils import get_uniform_random_unit_vector


B = [defaultdict(list) for i in range(N_S)]  # [{|x|: [{x, F(x), A}]}]

# X = [0, 1]^D is the design space


def constraints_func(x):  # TODO check this
    g1 = -x
    g2 = x-1

    return np.array([g1, g2])


def grad_constraints_func(x):
    g1_grad = np.full((x.shape), -1)
    g2_grad = np.full((x.shape), 1)
    return g1_grad, g2_grad


def g_alpha_func(alpha, x, _f_func):
    f = _f_func(x)
    return np.sum(alpha * f)


def minimize_g_alpha(x, _f_func):
    init_alpha = np.zeros(d)
    bounds = np.c_[np.full(d, 0), np.full(d, 1)]
    cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})
    sol = minimize(g_alpha_func, init_alpha, args=(x, _f_func), bounds=bounds, constraints=cons, method='SLSQP')
    if not sol.success:
        raise RuntimeError("Failed to solve")
    return sol.x


def get_A_mtx(x):
    f1_grad, f2_grad = grad_f_func(x)
    # g1_grad, g2_grad = constraints_gradient(x)
    # return np.array([f1_grad - f2_grad, g1_grad, g2_grad])
    return np.array([f1_grad - f2_grad])


def get_b_mtx(x):
    _, f2_grad = grad_f_func(x)
    return -f2_grad


def handle_err(A, y, b):
    err = np.linalg.norm(b - np.dot(A.T, y)) / np.linalg.norm(b)
    if abs(err) > 1e-2:  # TODO fix treshhold
        raise Exception('error is not close to 0')


def kk_solver_func(y, A, b):
    return np.linalg.norm(b - np.dot(A.T, y)) / np.linalg.norm(b)


def get_y_mtx(A, b):
    n, p = A.shape
    if n > p:
        y_bar = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
        handle_err(A, y_bar, b)
        return y_bar
    elif n < p:
        bounds = np.c_[np.full(n, 0), np.full(n, 1)]
        y_init = np.random.rand(n, 1)
        # TODO add constraints
        sol = minimize(kk_solver_func, y_init, args=(A, b), bounds=bounds, method='SLSQP')
        if not sol.success:
            raise RuntimeError("Failed to solve")
        return sol.x
    else:
        y_bar = np.dot(np.linalg.inv(A), b)
        handle_err(A, y_bar, b)
        return y_bar


def _select_min_sample(buffer_cell):
    min_dist = min(buffer_cell)
    x_j = random.choice(buffer_cell[min_dist])['x']

    return x_j


def stochastic_sampling(B):
    # uniformely select N_s buffer cells
    xs = []
    for j in range(N_S):
        if len(B) <= j or len(B[j]) == 0:
            x_j = np.random.rand(D)

        else:
            x_j = _select_min_sample(B[j])

        # uniform random unit vector that defines a stochastic direction (D dimensions)
        d_p = get_uniform_random_unit_vector(D)
        lambda_p = random.uniform(0, LAMBDA_P)
        x_s = x_j + 1.0/2**lambda_p * d_p
        # TODO clamp the result
        xs.append(x_s)

    return xs


def calculate_C(x_s):  # important for diversity
    return np.linalg.norm(x_s) * LAMBDA_S


# TODO fix this
def calculate_s(x_s):  # select random point within lambda_N dist (search direction)
    return get_uniform_random_unit_vector(d)


def calculate_z(x_s):
    return f_func(x_s) + calculate_s(x_s) * calculate_C(x_s)  # d dimensions the same as F(x)


def local_optimization_func(x, z):
    return np.linalg.norm(f_func(x) - z)


def get_x_o(x_s, z):
    bounds = np.c_[np.full(D, 0), np.full(D, 1)]
    sol = minimize(local_optimization_func, x_s, args=(z), bounds=bounds, method='SLSQP')
    if not sol.success:
        raise RuntimeError("Failed to solve")
    return sol.x


def local_optimization(x_s):
    z = calculate_z(x_s)
    x_o = get_x_o(x_s, z)
    return x_o


# def first_order_optimization(x_o):
#     A = get_A_mtx(x_o)
#     b = get_b_mtx(x_o)
#     y = get_y_mtx(A, b)
#     return y[:d-1]


def first_order_optimization(x_o):
    alpha = minimize_g_alpha(x_o, f_func)  # TODO fix this
    return run_homotopy(x_o, alpha, f_func, grad_f_func, grad2_f_func, D, 0, d)


def update_buffer(buffer_cell, x, A_i):
    x_dist = np.linalg.norm(x)

    size = 0
    for key in buffer_cell:
        size += len(buffer_cell[key])
    if size < K:
        buffer_cell[x_dist].append({'x': x, 'A': A_i})
        return buffer_cell, True

    max_dist = max(buffer_cell)
    if x_dist > max_dist:
        return buffer_cell, False

    buffer_cell[x_dist].append({'x': x, 'A': A_i})
    buffer_cell[max_dist] = buffer_cell[max_dist][:-1]
    return buffer_cell, True


def main():
    non_updated_cnt = 0
    while True:
        if non_updated_cnt == N_i:
            break
        xs = stochastic_sampling(B)
        for i, x_s in enumerate(xs):
            x_o = local_optimization(x_s)
            M_i = first_order_optimization(x_o)  # it should be D*d-1 matrix
            A_i = []
            for neighbour in M_i:
                A_i.append(f_func(neighbour))
            B[i], updated = update_buffer(B[i], x_s, A_i)  # TODO check x_o
            if updated:
                non_updated_cnt = 0
            else:
                non_updated_cnt += 1
