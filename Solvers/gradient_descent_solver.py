import numpy as np                      # Contains functions for linear algebra and numerical computation.
from Solvers import solver
import time


class GradientDescentSolver(solver.Solver):
    def solve(self, f, jacobian_f, gamma):
        xk = self.x0
        k = 0

        f_norm = np.linalg.norm(f(xk))

        xs = [xk]

        start_time = time.time()

        while (f_norm > self.tol) and (k < self.maxIts):
            xkp1 = xk - gamma * jacobian_f(xk).transpose() * f(xk)
            xk = xkp1
            f_norm = np.linalg.norm(f(xk))
            k += 1
            xs.append(xk)

        self.trace = xs
        self.its = k
        self.time_taken = time.time() - start_time
