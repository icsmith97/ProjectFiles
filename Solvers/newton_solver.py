import numpy as np                      # Contains functions for linear algebra and numerical computation.
import solver
import time


class NewtonSolver(solver.Solver):
    def solve(self, f, jacobian_f):
        xk = self.x0
        k = 0

        f_norm = np.linalg.norm(f(xk))

        xs = [xk]

        start_time = time.time()

        while (f_norm > self.tol) and (k < self.maxIts):
            xkp1 = xk - np.linalg.solve(jacobian_f(xk), f(xk))
            xk = xkp1
            xs.append(xk)
            f_norm = np.linalg.norm(f(xk))
            xs.append(xk)
            k += 1

        self.trace = xs
        self.its = k
        self.time_taken = time.time() - start_time
