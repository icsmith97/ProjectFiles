import numpy as np                      # Contains functions for linear algebra and numerical computation.
from Solvers import solver
import time


class GradientDescentSolver(solver.Solver):
    def solve(self, grad_f, hessian_f, gamma):
        xk = self.x0
        k = 0

        grad_f_norm = np.linalg.norm(grad_f(xk))

        xs = [xk]

        start_time = time.time()

        while (grad_f_norm > self.tol) and (k < self.maxIts):
            xkp1 = xk - gamma * (hessian_f(xk).transpose() @ grad_f(xk))
            xk = xkp1
            grad_f_norm = np.linalg.norm(grad_f(xk))
            k += 1
            xs.append(xk)

        self.trace = xs
        self.its = k
        self.time_taken = time.time() - start_time
