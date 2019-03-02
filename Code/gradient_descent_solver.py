import numpy as np
from Code import solver
import time
from Code import line_search


class GradDescSolver(solver.Solver):
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
            xk.reshape(n, 1)

            grad_f_norm = np.linalg.norm(grad_f(xk))
            xs = [xk]

            while (grad_f_norm > self.tol) and (k < self.maxIts):
                grad_fk = grad_f(xk)
                pk = - grad_fk
                alpha = line_search.backtracking(f, grad_fk, pk, xk)
                xkp1 = xk + alpha * pk
                xk = xkp1
                grad_f_norm = np.linalg.norm(grad_f(xk))
                k += 1
                xs.append(xk)

        self.trace = xs
        self.its = k

        if should_time:
            self.time_taken = time.time() - start
