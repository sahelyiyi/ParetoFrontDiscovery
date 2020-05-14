import random
import math
import numpy as np
from scipy.optimize import minimize

from utils import get_uniform_random_unit_vector

N_S = 10  # buffer size
K = 50
LAMBDA_P = 10
LAMBDA_S = 0.3
LAMBDA_B = 1e-2
LAMBDA_I = 1e-4
LAMBDA_N = 0.2 * N_S

D = 3
d = 2

BOUNDS = np.c_[np.full(D, 0), np.full(D, 1)]

B = []  # [{x, F(x), A}]

# X = [0, 1]^D is the design space


def design_constraints(x):  # TODO check this
    g1 = -x
    g2 = x-1

    return np.array([g1, g2])


def constraints_gradient(x):
    g1_grad = np.full((x.shape), -1)
    g2_grad = np.full((x.shape), 1)
    return g1_grad, g2_grad


def get_p(x):
    return 1 + 9 * np.sum(x[1:D] / (D - 1))


def get_f1(x):
    return x[0]


def ZDT1(x):
    f1 = get_f1(x)  # objective 1
    p = get_p(x)
    q = 1 - np.sqrt(f1 / p)
    f2 = p * q  # objective 2
    # f2 /= 10.0  # to scale the F

    return np.array([f1, f2])


def F(x):
    return ZDT1(x)


def F_gradient(x):
    f1 = get_f1(x)
    f1_grad = np.zeros((x.shape))
    f1_grad[0] = 1

    p = get_p(x)
    p_grad = np.full((x.shape), 9/(D-1))
    p_grad[0] = 0

    f2_grad = p_grad + ((p_grad * f1 + f1_grad * p) / (2 * math.sqrt(p * f1)))

    return f1_grad, f2_grad


def get_A_mtx(x):
    f1_grad, f2_grad = F_gradient(x)
    # g1_grad, g2_grad = constraints_gradient(x)
    # return np.array([f1_grad - f2_grad, g1_grad, g2_grad])
    return np.array([f1_grad - f2_grad])


def get_b_mtx(x):
    _, f2_grad = F_gradient(x)
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
    min_xs = buffer_cell[min_dist]
    x_j = random.choice(min_xs)['x']

    return x_j


def stochastic_sampling(B):
    # uniformely select N_s buffer cells
    xs = []
    for j in range(N_S):
        if len(B[j]) == 0:
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
    return F(x_s) + calculate_s(x_s) * calculate_C(x_s)  # d dimensions the same as F(x)


def local_optimization_func(x, z):
    return np.linalg.norm(ZDT1(x) - z)


def get_x_o(x_s, z):
    sol = minimize(local_optimization_func, x_s, args=(z), bounds=BOUNDS, method='SLSQP')
    if not sol.success:
        raise RuntimeError("Failed to solve")
    return sol.x


def local_optimization(x_s):
    z = calculate_z(x_s)
    x_o = get_x_o(x_s, z)
    return x_o


def first_order_optimization(x_o):
    A = get_A_mtx(x_o)
    b = get_A_mtx(x_o)
    y = get_y_mtx(A, b)
    return y[:d-1]


def update_buffer():
    pass


def main():
    while True:
        xs = stochastic_sampling(B)
        for x_s in xs:
            x_o = local_optimization(x_s)
            M = first_order_optimization(x_o)  # it should be D*d-1 matrix
