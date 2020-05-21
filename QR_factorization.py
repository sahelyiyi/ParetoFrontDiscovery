import numpy as np


def householder(x):
    alpha = x[0]
    s = np.power(np.linalg.norm(x[1:]), 2)
    v = x.copy()

    if s == 0:
        tau = 0
    else:
        t = np.sqrt(alpha**2 + s)
        v[0] = alpha - t if alpha <= 0 else -s / (alpha + t)

        tau = 2 * v[0]**2 / (s + v[0]**2)
        v /= v[0]

    return v.reshape(1, v.shape[0]), tau


def qr_factorization(A):
    m,n = A.shape
    R = A.copy()
    Q = np.identity(m)

    for j in range(0, n):
        # Apply Householder transformation.
        v, tau = householder(R[j:, j])
        H = np.identity(m)
        H[j:, j:] -= np.matmul(tau * v.reshape(-1, 1), v)
        R = np.matmul(H, R)
        Q = np.matmul(H, Q)

    return Q.T, R
