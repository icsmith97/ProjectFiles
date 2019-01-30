import numpy as np                      # Contains functions for linear algebra and numerical computation.
import time                             # Contains functions for timing code.
import matplotlib.pyplot as plt         # Contains functions to plot graphs.
from matplotlib import rc               # Allows access to the LaTeX interpreter for formatting graphs.
from Solvers import gradient_descent_solver
from Solvers import newton_solver
from Solvers import bfgs_solver
from Solvers import solver
from scipy.optimize import rosen, rosen_der, rosen_hess


def rosenbrock_f(x):
    return rosen(x)


def rosenbrock_grad_f(x):
    return rosen_der(x)


def rosenbrock_hess_f(x):
    return rosen_hess(x)

x0 = np.asarray([6, 6])

tol = 1e-6
max_its = 100000

solver = solver.Solver(x0, max_its, tol)

solver_gd = gradient_descent_solver.GradientDescentSolver(solver.x0, solver.maxIts, solver.tol)
solver_gd.solve(rosenbrock_f, rosenbrock_grad_f, Time=False)
print(solver_gd)

solver_nm = newton_solver.NewtonSolver(solver.x0, solver.maxIts, solver.tol)
solver_nm.solve(rosenbrock_grad_f, rosenbrock_hess_f, Time=False)
print(solver_nm)

solver_bfgs = bfgs_solver.BFGSSolver(solver.x0, solver.maxIts, solver.tol)
solver_bfgs.solve(rosenbrock_f, rosenbrock_grad_f, Time=False)
print(solver_bfgs)

root = np.asarray([1.0, 1.0])

gd_x, gd_y = solver_gd.error_pair_vectors(root)
nm_x, nm_y = solver_nm.error_pair_vectors(root)
nm_x, nm_y = nm_x[0:-2], nm_y[0:-2]
bf_x, bf_y = solver_bfgs.error_pair_vectors(root)

plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$\log e_k$', fontsize=14)
plt.ylabel(r'$\log e_{k+1}$', fontsize=14)
plt.grid(True)
plt.title(r'Log Error Plot', fontsize=16)
plt.plot(np.unique(gd_x), np.poly1d(np.polyfit(gd_x, gd_y, 1))(np.unique(gd_x)), 'r')
plt.plot(np.unique(nm_x), np.poly1d(np.polyfit(nm_x, nm_y, 1))(np.unique(nm_x)), 'b')
plt.plot(np.unique(bf_x), np.poly1d(np.polyfit(bf_x, bf_y, 1))(np.unique(bf_x)), 'g')
plt.legend(["Gradient Descent", "Newton's Method", "BFGS"])
plt.show()

slope_gd = np.polyfit(gd_x, gd_y, 1)[0]
slope_nm = np.polyfit(nm_x, nm_y, 1)[0]
slope_bf = np.polyfit(bf_x, bf_y, 1)[0]

print("Convergence Rate (GD): {:.3f}".format(slope_gd))
print("Convergence Rate (NM): {:.3f}".format(slope_nm))
print("Convergence Rate (BFGS): {:.3f}".format(slope_bf))
