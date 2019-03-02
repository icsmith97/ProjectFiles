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

x0 = np.repeat(3, n)

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

plot_region = [0, 3, 0, 3]

x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 9.6, 500)
x, y = np.meshgrid(x, y)

z = rosen(np.array([x, y]))

grad_pts = solver_gd.trace
newt_pts = solver_nm.trace
bfgs_pts = solver_bfgs.trace

grad_xs = []
grad_ys = []
newt_xs = []
newt_ys = []
bfgs_xs = []
bfgs_ys = []

for pt in grad_pts:
    grad_xs.append(pt[0])
    grad_ys.append(pt[1])

for pt in newt_pts:
    newt_xs.append(pt[0])
    newt_ys.append(pt[1])


for pt in bfgs_pts:
    bfgs_xs.append(pt[0])
    bfgs_ys.append(pt[1])

plt.figure(figsize=(10.0, 3.0))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.subplot(1, 3, 1)
plt.xlim(0.8, 3.2)
plt.ylim(0.8, 3.5)
plt.title(r'Gradient Descent', fontsize=16)
plt.contour(x, y, z, 500, linewidths=0.5)
plt.plot(grad_xs, grad_ys, 'xr-')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

plt.subplot(1, 3, 2)
plt.xlim(0.9, 3.2)
plt.ylim(-4, 9.5)
plt.title(r'Newton\textquotesingle s Method', fontsize=16)
plt.contour(x, y, z, 500, linewidths=0.5)
plt.plot(newt_xs, newt_ys, 'xb-')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

plt.subplot(1, 3, 3)
plt.xlim(0.8, 4)
plt.ylim(0.8, 4)
plt.title(r'BFGS', fontsize=16)
plt.contour(x, y, z, 500, linewidths=0.5)
plt.plot(bfgs_xs, bfgs_ys, 'xg-')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.tight_layout()
plt.show()

plt.savefig("ContourPlot.pdf", bbox_inches='tight')