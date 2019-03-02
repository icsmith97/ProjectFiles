import numpy as np
import matplotlib.pyplot as plt
from Code import gradient_descent_solver
from Code import newton_solver
from Code import bfgs_solver
from Code import solver
from scipy.optimize import rosen, rosen_der, rosen_hess


def rosenbrock_f(x):
    return rosen(x)


def rosenbrock_grad_f(x):
    return rosen_der(x)


def rosenbrock_hess_f(x):
    return rosen_hess(x)


n = 2

x0 = np.repeat(2, n)

tol = 1e-6
max_its = 100000

solver = solver.Solver(x0, max_its, tol)

solver_gd = gradient_descent_solver.GradDescSolver(solver.x0, solver.maxIts, solver.tol)
solver_gd.solve(rosenbrock_f, rosenbrock_grad_f, Time=False)
print(solver_gd)

solver_nm = newton_solver.NewtonSolver(solver.x0, solver.maxIts, solver.tol)
solver_nm.solve(rosenbrock_grad_f, rosenbrock_hess_f, Time=False)
print(solver_nm)

solver_bfgs = bfgs_solver.BFGSSolver(solver.x0, solver.maxIts, solver.tol)
solver_bfgs.solve(rosenbrock_f, rosenbrock_grad_f, Time=False)
print(solver_bfgs)

root = np.repeat(1.0, n)

gd_x, gd_y = solver_gd.error_pair_vectors(root)
nm_x, nm_y = solver_nm.error_pair_vectors(root)
bf_x, bf_y = solver_bfgs.error_pair_vectors(root)

f = plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$\log e_k$', fontsize=16)
plt.ylabel(r'$\log e_{k+1}$', fontsize=16)
plt.grid(True)

plt.plot(np.unique(gd_x), np.poly1d(np.polyfit(gd_x, gd_y, 1))(np.unique(gd_x)), 'r')
plt.plot(np.unique(nm_x), np.poly1d(np.polyfit(nm_x, nm_y, 1))(np.unique(nm_x)), 'b')
plt.plot(np.unique(bf_x), np.poly1d(np.polyfit(bf_x, bf_y, 1))(np.unique(bf_x)), 'g')

plt.plot(nm_x, nm_y, 'bx')
plt.plot(bf_x, bf_y, 'gx')

plt.legend(["Gradient Descent", "Newton's Method", "BFGS"], fontsize=16)
plt.show()

slope_gd = np.polyfit(gd_x, gd_y, 1)[0]
slope_nm = np.polyfit(nm_x, nm_y, 1)[0]
slope_bf = np.polyfit(bf_x, bf_y, 1)[0]

print("Convergence Rate (GD): {:.3f}".format(slope_gd))
print("Convergence Rate (NM): {:.3f}".format(slope_nm))
print("Convergence Rate (BFGS): {:.3f}".format(slope_bf))

f.savefig("RosenbrockConvergenceRate.pdf", bbox_inches='tight')
