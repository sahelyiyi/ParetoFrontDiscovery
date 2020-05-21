import math
import numpy as np
from scipy.optimize import minimize

from utils import get_uniform_positive_random_unit_vector
from QR_factorization import qr_decomposition


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


def minimize_g_alpha(alpha):
    init_x = np.zeros(N) # TODO check this
    bounds = np.c_[np.full(N, 0), np.full(N, 1)]
    sol = minimize(g_alpha_func, init_x, args=(alpha), bounds=bounds, method='Nelder-Mead')
    if not sol.success:
        raise RuntimeError("Failed to solve")
    return sol.x


def F_func(x, alpha):
    grad_f = grad_f_func(x)
    first_element = np.sum([alpha[i] * grad_f[i] for i in range(len(alpha))], axis=0)
    return np.concatenate((first_element , np.array([np.sum(alpha)-1])))


def hessian_of_the_lagrangian_func(x, alpha):
    grad2_f = grad2_f_func(x)
    return np.array([alpha[i] * grad2_f[i] for i in range(len(alpha))])


def F_jacobian_func(x, alpha):
    hess = hessian_of_the_lagrangian_func(x, alpha)
    hess_concat = np.concatenate((hess, np.zeros((hess.shape[0], 1))), axis=1)

    grad_f = grad_f_func(x)
    grad_f_concat = np.concatenate((grad_f, np.ones((grad_f.shape[0], 1))), axis=1)

    return np.concatenate((hess_concat, grad_f_concat), axis=0)


def phi_func(Q, epsilon, eta, x_star, alpha_star):
    res = np.dot(Q, np.concatenate((epsilon, eta)))
    res += np.concatenate((x_star, alpha_star))
    return res


def F_tilda_func(Q, epsilon, eta, x_star, alpha_star):
    res = phi_func(Q, epsilon, eta, x_star, alpha_star)
    x = res[:N]
    alpha = res[N:]
    return F_func(x, alpha)


def grad_F_tilda_func(Q, epsilon, eta, x_star, alpha_star):
    res = np.dot(F_jacobian_func(x_star, alpha_star).T, Q)
    return res[:, -(N+M+1):]


def newton_system(f, Df, Q, eta_0, epsilon, x_star, alpha_star, error=1e-1, max_iter=1000):  # TODO fix epsilon and max_iter
    eta_n = eta_0
    F_value = f(Q, epsilon, eta_n, x_star, alpha_star)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > error and iteration_counter < max_iter:
        delta = np.linalg.solve(Df(Q, epsilon, eta_n, x_star, alpha_star), -F_value)
        eta_n = eta_n + delta
        F_value = f(Q, epsilon, eta_n, x_star, alpha_star)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

    if abs(F_norm) > error:
        iteration_counter = -1
    return eta_n, iteration_counter


def run():
    # Step 1
    alpha = get_uniform_positive_random_unit_vector(K)
    x = minimize_g_alpha(alpha)
    # print('F(x, alpha) should be 0\n', F_func(x, alpha))

    # Step 2
    F_jacobian = F_jacobian_func(x, alpha)

    # Step 3
    Q_tilda, R = qr_decomposition(F_jacobian)
    R1 = R[:N+M+1, :]

    # Step 4
    s = 1e-10
    for j in range(len(R1)):
        if R1[j, j] < s:
            print('failed in checking R1')
            run()

    # Step 5
    Q_tilda_1 = Q_tilda[:, :N+M+1]
    Q_tilda_2 = Q_tilda[:, -(K-1):]
    Q = np.concatenate((Q_tilda_2, Q_tilda_1), axis=1)
    # (eq 14): Q = [q_1|...|q_(n+m+k)] --> q_i = Q[:, i]

    # Step 6
    I_size = K-1
    c = 1.0  # TODO check this
    epsilons = []
    for i in range(I_size):
        e_i = np.zeros(I_size)
        e_i[i] = 1
        q_i_bar = Q[:N, i]
        lambda_i = c / np.linalg.norm(grad_f_func(x) * q_i_bar)
        epsilon_i = lambda_i * e_i
        epsilons.append(epsilon_i)

    # Step 7
    result_points = []
    for epsilon_i in epsilons:

        # Step 10
        check_phi_alpha = False
        while not check_phi_alpha:
            # Step 9
            newton_iter = -1
            while newton_iter == -1:

                # Step 8
                eta_i = np.zeros(N+M+1)  # TODO check starting point
                newton_eta, newton_iter = newton_system(F_tilda_func, grad_F_tilda_func, Q, eta_i, epsilon_i, x, alpha)
                epsilon_i /= 2

            phi_alpha = phi_func(Q, epsilon_i, newton_eta, x, alpha)[N:]
            check_phi_alpha = False if len(phi_alpha[phi_alpha <= 0]) else True

        result_points.append(phi_func(Q, epsilon_i, newton_eta, x, alpha))

    return np.array(result_points)
