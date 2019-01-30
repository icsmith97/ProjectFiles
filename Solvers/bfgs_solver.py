import numpy as np                      # Contains functions for linear algebra and numerical computation.
import time                             # Contains functions for timing code.
from Solvers import solver


class BFGSSolver(solver.Solver):
    def solve(self, f, grad_f, **kwargs):

        should_time = kwargs.get('Time', False)
        repetitions = 1

        if (should_time):
            repetitions = 1000
            start = time.time()

        for i in range(repetitions):

            n = len(self.x0)  # use the starting value to determine the dimension of the problem
            xk = self.x0
            xk = xk.reshape(n, 1)
            hessian_k = np.eye(n)
            grad_norm = np.linalg.norm(grad_f(xk))
            k = 0
            xs = [xk]

            while grad_norm > self.tol:
                grad_fk = grad_f(xk)
                pk = - hessian_k @ grad_fk

                ak = self.backtracking_line_search(f, grad_fk, pk, xk)

                xkp1 = xk + ak * pk

                sk = xkp1 - xk

                grad_fkp1 = grad_f(xkp1)

                yk = grad_fkp1 - grad_fk

                # Nocedal 8.20 suggests rescaling h0 before performing the first BFGS update

                if k == 0:
                    hessian_k = np.inner(yk, sk) / np.inner(yk, yk) * np.eye(n)

                divisor = np.dot(yk.T, sk)

                if (divisor > 1e5):
                    rho_k = 1000.0
                else:
                    rho_k = 1.0 / divisor

                hessian_k = self.bfgs_update(hessian_k, rho_k, sk, yk, n)

                # prepare for the next iteration

                xk = xkp1
                xs.append(xk.T)
                grad_fk = grad_fkp1
                grad_norm = np.linalg.norm(grad_fk)
                k += 1

            # We return the approximated root, the time taken to find it, and the number of
            # iterations taken to find it

            self.trace = xs
            self.its = k

        if (should_time):
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

    @staticmethod
    def bfgs_update(hessian_k, rho_k, sk, yk, n):
        rsy = (rho_k * sk) @ yk.T
        rys = (rho_k * yk) @ sk.T
        rss = (rho_k * sk) @ sk.T

        hkp1 = (np.eye(n) - rsy) @ hessian_k @ (np.eye(n) - rys) + rss

        return hkp1
