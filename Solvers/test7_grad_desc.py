import my_bfgs
import numpy as np

def f(x):
    return x**4 - 3*x**3 + 2


def fp(x):
    return 4*x**3 - 9*x**2

root = 9/4

x0 = 6

tol = 1e-8
max_its = 1000
gamma = 0.01

print(my_bfgs.gradient_descent(x0, f, fp, tol, max_its, gamma, root ))

def g(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2


def gp(x):
    return np.array([2 * x[0], 2 * x[1], 2 * x[2], 3 * x[3]]).reshape(1, 4)

root = np.array([0, 0, 0, 0]).reshape(4, 1)

x0 = np.array([43, 12, 105, -32]).reshape(4, 1)

print(my_bfgs.gradient_descent_nd(x0, g, gp, tol, max_its, gamma, root ))