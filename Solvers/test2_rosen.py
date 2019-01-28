from Solvers import solver
from Solvers import comparisons
import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess

"""
Test 2 :    Rosenbrock Function (n = 2)
            Global Minimum is [1, 1]

            f(x)  = rosen(x)
            grad_f(x) = rosen_der(x)
            hess_f(x) = rosen_hes(x)
"""


def rosenbrock_f(x):
    return rosen(x)


def rosenbrock_grad_f(x):
    return rosen_der(x)


def rosenbrock_hess_f(x):
    return rosen_hess(x)


x0 = np.array([2, 2]).T
tol = 1e-6
max_its = 100000

solver = solver.Solver(x0, max_its, tol)
comparisons.compare(solver, rosenbrock_f, rosenbrock_grad_f, rosenbrock_hess_f)

root = np.array([1, 1]).T
