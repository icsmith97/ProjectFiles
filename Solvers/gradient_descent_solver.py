import numpy as np                      # Contains functions for linear algebra and numerical computation.
from Solvers import solver
import time


class GradientDescentSolver(solver.Solver):
    def solve(self, f, grad_f, **kwargs):
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
            xs = [xk]

            while (grad_f_norm > self.tol) and (k < self.maxIts):
                grad_fk = grad_f(xk)
                pk = - grad_fk
                alpha = self.backtracking_line_search(f, grad_fk, pk, xk)
                xkp1 = xk + alpha * pk
                xk = xkp1
                grad_f_norm = np.linalg.norm(grad_f(xk))
                k += 1
                xs.append(xk)

        self.trace = xs
        self.its = k

        if should_time:
            self.time_taken = time.time() - start

    @staticmethod
    def backtracking_line_search(f, grad_fk, pk, xk):
        a = 1
        rho = 0.5
        c = 0.1

        # Starting with a = 1, then chose values for rho and c in (0,1).

        while f(xk + (a * pk)) > (f(xk) + (c * a) * (grad_fk.transpose() @ pk)):
            a = rho * a

        return a
