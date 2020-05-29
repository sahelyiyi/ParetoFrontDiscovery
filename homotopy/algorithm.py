import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from homotopy.numerical_result_1 import f_func, grad_f_func, grad2_f_func
from utils import get_uniform_positive_random_unit_vector
from QR_factorization import qr_factorization


N = 2
M = 0
K = 2


def g_alpha_func(x, alpha, _f_func):
    f = _f_func(x)
    return np.sum(alpha * f)


def minimize_g_alpha(alpha, _f_func, n):
    init_x = np.zeros(n)  # TODO check this
    bounds = np.c_[np.full(n, 0), np.full(n, 1)]
    sol = minimize(g_alpha_func, init_x, args=(alpha, _f_func), bounds=bounds, method='Nelder-Mead')
    if not sol.success:
        raise RuntimeError("Failed to solve")
    return sol.x


def F_func(x, alpha, _grad_f_func):
    grad_f = _grad_f_func(x)
    first_element = np.sum([alpha[i] * grad_f[i] for i in range(len(alpha))], axis=0)
    return np.concatenate((first_element, np.array([np.sum(alpha)-1])))


def hessian_of_the_lagrangian_func(x, alpha, _grad2_f_func):
    grad2_f = _grad2_f_func(x)
    res = np.array([alpha[i] * grad2_f[i] for i in range(len(alpha))])
    return np.sum(res, axis=0)


def F_jacobian_func(x, alpha, _grad_f_func, _grad2_f_func):
    hess = hessian_of_the_lagrangian_func(x, alpha, _grad2_f_func)
    hess_concat = np.concatenate((hess, np.zeros((hess.shape[0], 1))), axis=1)

    grad_f = _grad_f_func(x)
    grad_f_concat = np.concatenate((grad_f, np.ones((grad_f.shape[0], 1))), axis=1)

    return np.concatenate((hess_concat, grad_f_concat), axis=0)  # (n+m+k)*(n+m+1)


def phi_func(Q, epsilon, eta, x_star, alpha_star):
    res = np.dot(Q, np.concatenate((epsilon, eta)))
    res += np.concatenate((x_star, alpha_star))
    return res


def F_tilda_func(Q, epsilon, eta, x_star, alpha_star, _grad_f_func, n):
    res = phi_func(Q, epsilon, eta, x_star, alpha_star)
    x = res[:n]
    alpha = res[n:]
    return F_func(x, alpha, _grad_f_func)


def grad_F_tilda_func(Q, x_star, alpha_star, _grad_f_func, _grad2_f_func, n, m):
    res = np.dot(F_jacobian_func(x_star, alpha_star, _grad_f_func, _grad2_f_func).T, Q)
    return res[:, -(n+m+1):]


def plot_steps_chart(xs, x_star):
    X = []
    Y = []
    for item in xs:
        f1, f2 = f_func(item)
        X.append(f1)
        Y.append(f2)

    plt.plot(X, Y, 'bo')
    f1, f2 = f_func(x_star)
    X = [f1]
    Y = [f2]
    plt.plot(X, Y, 'ro')
    plt.axis([-0.5, 1.5, -0.5, 1.5])
    plt.savefig('results/candidate_set.jpg')


def newton_system(f, Df, Q, eta_0, epsilon, x_star, alpha_star, _grad_f_func, _grad2_f_func, n, m, plot=False, error=1e-1, max_iter=1000):  # TODO fix epsilon and max_iter
    eta_n = eta_0
    f_value = f(Q, epsilon, eta_n, x_star, alpha_star, _grad_f_func, n)
    f_norm = np.linalg.norm(f_value, ord=2)  # l2 norm of vector
    iteration_counter = 0

    xs = []

    while abs(f_norm) > error and iteration_counter < max_iter:
        df_value = Df(Q, x_star, alpha_star, _grad_f_func, _grad2_f_func, n, m)
        delta = np.linalg.solve(df_value, -f_value)
        eta_n = eta_n + delta
        f_value = f(Q, epsilon, eta_n, x_star, alpha_star, _grad_f_func, n)
        f_norm = np.linalg.norm(f_value, ord=2)

        xs.append(phi_func(Q, epsilon, eta_n, x_star, alpha_star)[:n])

        iteration_counter += 1

    if abs(f_norm) > error:
        iteration_counter = -1
    elif plot:
        plot_steps_chart(xs, x_star)
    return eta_n, iteration_counter


def run_homotopy(x, alpha, _f_func, _grad_f_func, _grad2_f_func, n, m, k):
    # Step 2
    F_jacobian = F_jacobian_func(x, alpha, _grad_f_func, _grad2_f_func)

    # Step 3
    Q_tilda, R = qr_factorization(F_jacobian)
    R1 = R[:n+m+1, :]

    # Step 4
    s = 1e-20
    for j in range(len(R1)):
        if R1[j, j] < s:
            print('failed in checking R1')
            run()

    # Step 5
    Q_tilda_1 = Q_tilda[:, :n+m+1]
    Q_tilda_2 = Q_tilda[:, -(k-1):]
    Q = np.concatenate((Q_tilda_2, Q_tilda_1), axis=1)
    # (eq 14): Q = [q_1|...|q_(n+m+k)] --> q_i = Q[:, i]

    # Step 6
    I_size = k-1
    c = 1.0  # TODO check this
    epsilons = []
    for i in range(I_size):
        e_i = np.zeros(I_size)
        e_i[i] = 1
        q_i_bar = Q[:n, i]
        lambda_i = c / np.linalg.norm(_grad_f_func(x) * q_i_bar)
        epsilon_i = lambda_i * e_i
        epsilons.append(epsilon_i)

    # Step 7
    result_points = []
    for epsilon_i in epsilons:
        # epsilon_i = np.array([0.06])

        # Step 10
        check_phi_alpha = False
        while not check_phi_alpha:
            # Step 9
            newton_iter = -1
            while newton_iter == -1:

                # Step 8
                eta_i = np.zeros(n+m+1)  # TODO check starting point
                newton_eta, newton_iter = newton_system(F_tilda_func, grad_F_tilda_func, Q, eta_i, epsilon_i, x, alpha, _grad_f_func, _grad2_f_func, n, m)
                epsilon_i /= 2

            phi_alpha = phi_func(Q, epsilon_i, newton_eta, x, alpha)[n:]
            check_phi_alpha = False if len(phi_alpha[phi_alpha <= 0]) else True

        result_points.append(phi_func(Q, epsilon_i, newton_eta, x, alpha)[:n])

    return np.array(result_points)


def run(_f_func=f_func, _grad_f_func=grad_f_func, _grad2_f_func=grad2_f_func, n=N, m=M, k=K):
    # Step 1
    alpha = get_uniform_positive_random_unit_vector(k)
    x = minimize_g_alpha(alpha, _f_func, n)
    # print('F(x, alpha) should be 0\n', F_func(x, alpha, _grad_f_func))

    return run_homotopy(x, alpha, _f_func, _grad_f_func, _grad2_f_func, n, m, k)
