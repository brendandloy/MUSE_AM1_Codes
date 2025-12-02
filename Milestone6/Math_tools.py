from numpy import zeros
from numpy.linalg import norm
from scipy.linalg import solve


def derivative(f, x, r, h=1e-7):
    """
    The tolerance "h" is optional. By default, h=1e-7
    """

    return (f(x + r*h) - f(x - r*h)) / (2 * h)


def Jacobian(f, x):

    J = zeros((len(x), len(x)), dtype=complex)

    for j in range(len(x)):
        r = zeros((len(x)), dtype=complex)
        r[j] = 1
        J[:, j] = derivative(f, x, r)

    return J


def Gauss(A, b):

    return solve(A, b)


def Newton(f, x0):

    x = x0
    Dx = 1e-3

    while norm(Dx) > 1e-10:
        A = Jacobian(f, x)
        Dx = Gauss(A, -f(x))
        x = x + Dx

    return x