import numpy as np
from Code import bfgs_solver
from scipy.optimize import rosen, rosen_der


def rosenbrock_f(x):
    return rosen(x)


def rosenbrock_grad_f(x):
    return rosen_der(x)

n = 2000

x0 = np.repeat(1.5, n)

tol = 1e-1
max_its = 10000

dimensions = "Number of Dimensions: {}\n".format(n)
print(dimensions)

solver_bfgs = bfgs_solver.BFGSSolver(x0, max_its, tol)
solver_bfgs.solve(rosenbrock_f, rosenbrock_grad_f, Time=False)
print(solver_bfgs)

n+=100