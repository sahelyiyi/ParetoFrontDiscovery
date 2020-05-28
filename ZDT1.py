import numpy as np
import math
from constants import D


def p_func(x):
    shape = x.shape[0]
    return 1 + 9 * np.sum(x[1:shape] / (shape - 1))


def grad_p_func(x):
    shape = x.shape[0]
    grad_p = np.full((shape), 9 / (shape - 1))
    grad_p[0] = 0
    return grad_p


def grad2_p_func(x):
    # return np.zeros((x.shape[0], x.shape[0]))  # TODO check this
    return np.random.rand(x.shape[0], x.shape[0])


def f1_func(x):
    return x[0]


def grad_f1_func(x):
    grad_f1 = np.zeros((x.shape))
    grad_f1[0] = 1
    return grad_f1


def grad2_f1_func(x):
    # return np.zeros((x.shape[0], x.shape[0]))  # TODO check this
    return np.random.rand(x.shape[0], x.shape[0])


def grad_pf1_func(x):
    f1 = f1_func(x)
    grad_f1 = grad_f1_func(x)

    p = p_func(x)
    grad_p = grad_p_func(x)

    return grad_p * f1 + grad_f1 * p


def grad2_pf1_func(x):
    f1 = f1_func(x)
    grad_f1 = grad_f1_func(x)
    grad2_f1 = grad2_f1_func(x)

    p = p_func(x)
    grad_p = grad_p_func(x)
    grad2_p = grad2_p_func(x)
    return grad2_p * f1 + 2 * grad_p * grad_f1 + grad2_f1 * p


def grad_pf1sqrt(x):
    f1 = f1_func(x)
    p = p_func(x)

    # return grad_pf1_func(x) / (2 * math.sqrt(p * f1))  # TODO check this
    return grad_pf1_func(x) / (2 * math.sqrt(abs(p * f1)))  # TODO fix this


def grad2_pf1sqrt(x):
    f1 = f1_func(x)
    p = p_func(x)

    return (grad2_pf1_func(x) * math.sqrt(p * f1) - grad_pf1_func(x) * grad_pf1sqrt(x)) / (2 * p * f1)


def f2_func(x):
    f1 = f1_func(x)
    p = p_func(x)
    q = 1 - np.sqrt(f1 / p)
    f2 = p * q
    return f2


def grad_f2_func(x):
    return grad_p_func(x) + grad_pf1sqrt(x)


def grad2_f2_func(x):
    return grad2_p_func(x) + grad2_pf1sqrt(x)


def ZDT1_func(x):
    if x.shape[0] != D:
        raise Exception('invalid shape of x.')
    f1 = f1_func(x)  # objective 1
    f2 = f2_func(x)  # objective 2
    # f2 /= 10.0  # to scale the F  # TODO check this

    return np.array([f1, f2])


def f_func(x):
    return ZDT1_func(x)


def grad_f_func(x):
    grad_f1 = grad_f1_func(x)
    grad_f2 = grad_f2_func(x)
    return np.array([grad_f1, grad_f2])


def grad2_f_func(x):
    grad2_f1 = grad2_f1_func(x)
    grad2_f2 = grad2_f2_func(x)
    return np.array([grad2_f1, grad2_f2])
