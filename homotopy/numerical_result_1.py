import math
import numpy as np

A_C = 45
A_1 = 40
A_2 = 25
D = 0.5


def a_func(x):
    return (2 * math.pi / 360) * (A_C + A_1 * math.sin(2 * math.pi * x[0]) + A_2 * math.sin(2 * math.pi * x[1]))


def grad_a_func(x):
    grad_x1 = (2 * math.pi / 360) * A_1 * 2 * math.pi * math.cos(2 * math.pi * x[0])
    grad_x2 = (2 * math.pi / 360) * A_2 * 2 * math.pi * math.cos(2 * math.pi * x[1])
    return np.array([grad_x1, grad_x2])


def grad2_a_func(x):
    grad2_x1 = (2 * math.pi / 360) * A_1 * (2 * math.pi)**2 * -math.sin(2 * math.pi * x[0])
    grad2_x2 = (2 * math.pi / 360) * A_2 * (2 * math.pi)**2 * -math.sin(2 * math.pi * x[1])
    grad2_x1_x2 = 0
    return np.array([[grad2_x1, grad2_x1_x2], [grad2_x1_x2, grad2_x2]])


def b_func(x):
    return 1 + D * math.cos(2 * math.pi * x[1])


def grad_b_func(x):
    grad_x1 = - D * 2 * math.pi * math.sin(2 * math.pi * x[1])
    grad_x2 = 0
    return np.array([grad_x1, grad_x2])


def grad2_b_func(x):
    grad2_x1 = - D * (2 * math.pi)**2 * math.cos(2 * math.pi * x[1])
    grad2_x2 = 0
    grad2_x1_x2 = 0
    return np.array([[grad2_x1, grad2_x1_x2], [grad2_x1_x2, grad2_x2]])


def f1_func(x):
    return math.cos(a_func(x)) * b_func(x)


def grad_f1_func(x):
    a = a_func(x)
    grad_a = grad_a_func(x)

    b = b_func(x)
    grad_b = grad_b_func(x)

    return grad_a * -math.sin(a) * b + math.cos(a) * grad_b


def grad2_f1_func(x):
    a = a_func(x)
    grad_a = grad_a_func(x)
    grad2_a = grad2_a_func(x)

    b = b_func(x)
    grad_b = grad_b_func(x)
    grad2_b = grad2_b_func(x)

    return grad2_a * -math.sin(a) * b + grad_a * (grad_a * -math.cos(a) * b + 2 * -math.sin(a) * grad_b) + math.cos(a) * grad2_b


def f2_func(x):
    return math.sin(a_func(x)) * b_func(x)


def grad_f2_func(x):
    a = a_func(x)
    grad_a = grad_a_func(x)

    b = b_func(x)
    grad_b = grad_b_func(x)

    return grad_a * math.cos(a) * b + math.sin(a) * grad_b


def grad2_f2_func(x):
    a = a_func(x)
    grad_a = grad_a_func(x)
    grad2_a = grad2_a_func(x)

    b = b_func(x)
    grad_b = grad_b_func(x)
    grad2_b = grad2_b_func(x)

    return grad2_a * math.cos(a) * b + grad_a * (grad_a * -math.sin(a) * b + 2 * math.cos(a) * grad_b) + math.sin(a) * grad2_b


def f_func(x):
    f1 = f1_func(x)
    f2 = f2_func(x)
    return np.array([f1, f2])


def grad_f_func(x):
    grad_f1 = grad_f1_func(x)
    grad_f2 = grad_f2_func(x)
    return np.array([grad_f1, grad_f2])


def grad2_f_func(x):
    grad2_f1 = grad2_f1_func(x)
    grad2_f2 = grad2_f2_func(x)
    return np.array([grad2_f1, grad2_f2])
