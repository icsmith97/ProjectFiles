from Solvers import solver
from Solvers import comparisons
import numpy as np

"""
Test 1 :    Simple Function (Sphere Function for n = 3)
            Global Minimum is [0, 0, 0]

            f(x)  = x[0]^2 + x[1]^2 + x[2]^2
            f'(x) = [2*x[0], 2*x[1], 2*x[2]]
"""


def sphere(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


def sphere_grad(x):
    return np.array([2 * x[0], 2 * x[1], 2 * x[2]])


def sphere_hessian(x):
    return np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])


x0 = np.array([2, 2, 2]).T
max_its = 1000
tol = 1e-4

solver = solver.Solver(x0, max_its, tol)
comparisons.compare(solver, sphere, sphere_grad, sphere_hessian)

root = np.array([0, 0, 0]).T

