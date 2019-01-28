from Solvers import gradient_descent_solver
from Solvers import newton_solver
from Solvers import bfgs_solver
import numpy as np

def compare(solver, test_f, test_grad_f, test_hessian_f, gamma):
    solver_gd = gradient_descent_solver.GradientDescentSolver(solver.x0, solver.maxIts, solver.tol)
    solver_gd.solve(test_grad_f, gamma, Time=False)
    print(solver_gd)

    solver_nm = newton_solver.NewtonSolver(solver.x0, solver.maxIts, solver.tol)
    solver_nm.solve(test_grad_f, test_hessian_f, Time=False)
    print(solver_nm)

    solver_bfgs = bfgs_solver.BFGSSolver(solver.x0, solver.maxIts, solver.tol)
    solver_bfgs.solve(test_f, test_grad_f, Time=False)
    print(solver_bfgs)
