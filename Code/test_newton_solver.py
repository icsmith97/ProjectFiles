import newton_solver
import numpy as np

"""
Test 1 :    Simple Function (Sphere Function for n = 4)
            Global Minimum is [0, 0, 0, 0]

            f(x)  = x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2
            f'(x) = [2*x[0], 2*x[1], 2*x[2], 2*x[3]]
"""


def f(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2


def fp(x):
    return np.array([2 * x[0], 2 * x[1], 2 * x[2], 3 * x[3]]).reshape(4, 1)


n = 4
root = np.array([0, 0, 0, 0]).reshape(4, 1)

x0 = np.array([43, 12, 105, -32]).reshape(4, 1)

tol = 1e-8
max_its = 1000

my_solver = newton_solver.NewtonSolver(x0, max_its, tol)
my_solver.solve(f, fp)


