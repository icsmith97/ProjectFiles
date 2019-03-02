from Code import gradient_descent_solver
import numpy as np


#def f(x):
 #   return x[0]**3 - x[0]**2 + x[0] + x[0] * x[1]


#def fp(x):
  #  return np.array([3*x[0]**2 - 2*x[0] + 1 + x[1], x[0]], dtype=float)

def f(x):
    return 2*x[0]**2 + x[1]**3 - 5

def fp(x):
    return np.array([4*x[0], 3*x[1]**2])


x0 = np.array([1, 1])
maxIts = 20
tol = 1e-2


solver_gd = gradient_descent_solver.GradDescSolver(x0, maxIts, tol)
solver_gd.solve(f, fp, Time=False)
print(solver_gd)
