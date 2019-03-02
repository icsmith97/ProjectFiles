import numpy as np
from Code import bfgs_solver
from scipy.optimize import rosen, rosen_der
import time
import matplotlib.pyplot as plt


def rosenbrock_f(x):
    return rosen(x)


def rosenbrock_grad_f(x):
    return rosen_der(x)

ns = []
times = []

n = 5

while(n <= 505):

    x0 = np.repeat(1.5, n)

    tol = 1e-4
    max_its = 10000

    start = time.time()
    solver_bfgs = bfgs_solver.BFGSSolver(x0, max_its, tol)
    solver_bfgs.solve(rosenbrock_f, rosenbrock_grad_f, Time=False)
    time_taken = time.time() - start

    if (n - 5) % 20  == 0:
        output = "Finished n = {}\n".format(n)
        print(output)
    ns.append(n)
    times.append(time_taken)
    n += 10

f = plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'Dimension $n$', fontsize=16)
plt.ylabel(r'Runtime ($s$)', fontsize=16)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.grid(True)
plt.plot(ns, times, 'r')
plt.show()

f.savefig("RunTimes.pdf", bbox_inches='tight')

g = plt.figure(2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'Log Dimension', fontsize=16)
plt.ylabel(r'Log Runtime', fontsize=16)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.grid(True)
plt.plot(np.log(ns), np.log(times), 'r')
plt.show()

g.savefig("LogRunTimes.pdf", bbox_inches='tight')