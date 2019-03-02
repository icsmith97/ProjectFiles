import numpy as np
from Code import solver
import time


class NewtonSolver(solver.Solver):
    def solve(self, grad_f, hessian_f, **kwargs):
        should_time = kwargs.get('Time', False)
        repetitions = 1

        if should_time:
            repetitions = 1000
            start = time.time()

        for i in range(repetitions):

            n = len(self.x0)
            xk = self.x0
            k = 0

            grad_f_norm = np.linalg.norm(grad_f(xk))

            xs = [xk.reshape(n, 1)]

            while (grad_f_norm > self.tol) and (k < self.maxIts):
                if len(xk) > 1:
                    xkp1 = xk - np.linalg.solve(hessian_f(xk), grad_f(xk))
                xk = xkp1
                xs.append(xk.reshape(n, 1))
                grad_f_norm = np.linalg.norm(grad_f(xk))
                k += 1

        self.trace = xs
        self.its = k

        if should_time:
            self.time_taken = time.time() - start
