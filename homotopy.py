import math
import numpy as np
from scipy.optimize import minimize

from utils import get_uniform_random_unit_vector


A_C = 45
A_1 = 40
A_2 = 25
D = 0.5

N = 2
M = 0
K = 2


def a_func(x):
    return (2 * math.pi / 360) * (A_C + A_1 * math.sin(2*math.pi*x[0]) + A_2 * math.sin(2*math.pi*x[1]))


def grad_a_func(x):
    grad_x1 = (2 * math.pi / 360) * A_1 * 2*math.pi * math.cos(2*math.pi*x[0])
    grad_x2 = (2 * math.pi / 360) * A_2 * 2*math.pi * math.cos(2*math.pi*x[1])
    return np.array([grad_x1, grad_x2])


def grad2_a_func(x):
    grad_x1 = (2 * math.pi / 360) * A_1 * (2*math.pi)**2 * -math.sin(2*math.pi*x[0])
    grad_x2 = (2 * math.pi / 360) * A_2 * (2*math.pi)**2 * -math.sin(2*math.pi*x[1])
    return np.array([grad_x1, grad_x2])


def b_func(x):
    return 1 + D * math.cos(2*math.pi*x[1])


def grad_b_func(x):
    grad_x1 = 0
    grad_x2 = - D * 2*math.pi * math.sin(2*math.pi*x[1])
    return np.array([grad_x1, grad_x2])


def grad2_b_func(x):
    grad_x1 = 0
    grad_x2 = - D * (2*math.pi)**2 * math.cos(2*math.pi*x[1])
    return np.array([grad_x1, grad_x2])


def f_func(x):
    f1 = math.cos(a_func(x)) * b_func(x)
    f2 = math.sin(a_func(x)) * b_func(x)
    return np.array([f1, f2])


def grad_f_func(x):
    a = a_func(x)
    grad_a = grad_a_func(x)

    b = b_func(x)
    grad_b = grad_b_func(x)

    grad_f1 = grad_a * -math.sin(a) * b + math.cos(a) * grad_b
    grad_f2 = grad_a * math.cos(a) * b + math.sin(a) * grad_b
    return np.array([grad_f1, grad_f2])


def grad2_f_func(x):
    a = a_func(x)
    grad_a = grad_a_func(x)
    grad2_a = grad2_a_func(x)

    b = b_func(x)
    grad_b = grad_b_func(x)
    grad2_b = grad2_b_func(x)

    grad2_f1 = grad2_a * -math.sin(a) * b + grad_a * (grad_a * -math.cos(a) * b + 2 * -math.sin(a) * grad_b) + math.cos(a) * grad2_b
    grad2_f2 = grad2_a * math.cos(a) * b + grad_a * (grad_a * -math.sin(a) * b + 2 * math.cos(a) * grad_b) + math.sin(a) * grad2_b
    return np.array([grad2_f1, grad2_f2])


def g_alpha_func(x, alpha):
    f = f_func(x)
    return np.sum(alpha * f)


def hessian_of_the_lagrangian_func(x, alpha):
    grad2_f = grad2_f_func(x)
    return np.array([alpha[i] * grad2_f[i] for i in range(len(alpha))])


def F_jacobian_func(x, alpha):
    hess = hessian_of_the_lagrangian_func(x, alpha)
    hess_concat = np.concatenate((hess, np.zeros((hess.shape[0], 1))), axis=1)

    grad_f = grad_f_func(x)
    grad_f_concat = np.concatenate((grad_f, np.ones((grad_f.shape[0], 1))), axis=1)

    return np.concatenate((hess_concat, grad_f_concat), axis=0)


def minimize_g_alpha(alpha):
    init_x = np.zeros(N)
    bounds = np.c_[np.full(N, 0), np.full(N, 1)]
    sol = minimize(g_alpha_func, init_x, args=(alpha), bounds=bounds, method='Nelder-Mead')
    if not sol.success:
        raise RuntimeError("Failed to solve")
    return sol.x


def run():
    # Step 1
    alpha = get_uniform_random_unit_vector(K)
    x = minimize_g_alpha(alpha)

    # Step 2
    F_jacobian = F_jacobian_func(x, alpha)

    # Step 3
    Q_tilda, R = np.linalg.qr(F_jacobian, mode='complete')
    R1 = R[:N+M+1, :]

    # Step 4
    s = 1e-10
    for j in range(len(R1)):
        if R1[j, j] < s:
            run()

    # Step 5
    Q_tilda_1 = Q_tilda[:, :N+M+1]
    Q_tilda_2 = Q_tilda[:, -(K-1):]
    Q = np.concatenate((Q_tilda_2, Q_tilda_1), axis=1)
    
    # Step 6
