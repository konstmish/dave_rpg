import numpy as np
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix

supported_penalties = ['l1']

def f(x, A, b, l2):
    l = np.log(1 + np.exp(-A.dot(x).multiply(b).A))
    # l = np.log(1 + np.exp(-A.dot(x).T * b))
    m = b.shape[0]
    return np.sum(l) / m + l2/2 * norm(x) ** 2

def f_grad(x, A, b, l2):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert l2 >= 0
    denom = csr_matrix(1/(1 + np.exp(A.dot(x).multiply(b).A)))
    g = -(A.multiply(b).multiply(denom).sum(axis=0).T)
    m = b.shape[0]
    return csr_matrix(g) / m + l2 * x

def r(x, l1):
    return l1 * norm(x, ord = 1)

def F(x, A, b, l2, l1):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert ((l2 >= 0) & (l1 >= 0))
    return f(x, A, b, l2) + r(x, l1)

def prox_r(x, gamma, coef, penalty = 'l1'):
    assert penalty in supported_penalties
    assert(gamma > 0 and coef >= 0)
    if penalty == 'l1':
        l1 = coef
        return x - abs(x).minimum(l1 * gamma).multiply(x.sign())