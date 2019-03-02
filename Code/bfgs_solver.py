import numpy as np
import time
from Code import solver
from Code import line_search


class BFGSSolver(solver.Solver):
    def solve(self, f, grad_f, **kwargs):

        should_time = kwargs.get('Time', False)
        repetitions = 1

        if should_time:
            repetitions = 1000
            start = time.time()

        for i in range(repetitions):

            n = len(self.x0)
            xk = self.x0
            xk = xk.reshape(n, 1)

            xs = [xk]
            hessian_k = np.eye(n)
            grad_norm = np.linalg.norm(grad_f(xk))
            k = 0

            while grad_norm > self.tol and k < self.maxIts:
                grad_fk = grad_f(xk)
                pk = - hessian_k @ grad_fk

                ak = line_search.backtracking(f, grad_fk, pk, xk)

                xkp1 = xk + ak * pk

                sk = xkp1 - xk

                grad_fkp1 = grad_f(xkp1)

                yk = grad_fkp1 - grad_fk

                if k == 0:
                    hessian_k = np.inner(yk, sk) / np.inner(yk, yk) * np.eye(n)

                divisor = np.dot(yk.T, sk)

                if divisor < 1e-12:
                    rho_k = 1000
                else:
                    rho_k = 1.0 / divisor

                hessian_k = self.bfgs_update(hessian_k, rho_k, sk, yk, n)

                xk = xkp1
                xs.append(xk)
                grad_fk = grad_fkp1
                grad_norm = np.linalg.norm(grad_fk)
                k += 1

            self.trace = xs
            self.its = k

        if should_time:
            self.time_taken = time.time() - start

    @staticmethod
    def bfgs_update(hessian_k, rho_k, sk, yk, n):
        rsy = (rho_k * sk) @ yk.T
        rys = (rho_k * yk) @ sk.T
        rss = (rho_k * sk) @ sk.T

        hkp1 = (np.eye(n) - rsy) @ hessian_k @ (np.eye(n) - rys) + rss

        return hkp1
