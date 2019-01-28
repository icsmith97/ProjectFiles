import numpy as np                      # Contains functions for linear algebra and numerical computation.
from Solvers import solver
import time


class GradientDescentSolver(solver.Solver):
    def solve(self, grad_f, gamma, **kwargs):
        should_time = kwargs.get('Time', False)
        repetitions = 1

        if (should_time):
            repetitions = 1000
            start = time.time()

        for i in range(repetitions):


            n = len(self.x0)  # use the starting value to determine the dimension of the problem
            xk = self.x0
            k = 0

            grad_f_norm = np.linalg.norm(grad_f(xk))
            xs = [xk]

            start_time = time.time()

            while (grad_f_norm > self.tol) and (k < self.maxIts):
                xkp1 = xk - gamma * grad_f(xk)
                xk = xkp1
                grad_f_norm = np.linalg.norm(grad_f(xk))
                k += 1
                xs.append(xk)

        self.trace = xs
        self.its = k

        if (should_time):
            self.time_taken = time.time() - start