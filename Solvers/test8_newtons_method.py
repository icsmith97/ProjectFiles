import my_bfgs
import numpy as np

def f(x):
    return np.array([x[0]**2 + x[1]**2 - 2*x[0]*x[1] - 1, x[0]**2 - x[1]**2 - 7]).reshape(2, 1)

def j_f(x):
    return np.array(
        [[2*x[0] - 2*x[1],   2*x[1] - 2*x[0]],
         [2*x[0]         ,   -2*x[1]]]
    ).reshape(2, 2)

tol = 1e-8
max_its = 1000
root = np.array([4, 3])
x0 = np.array([1, -1]).reshape(2, 1)

print(my_bfgs.newtons_method_nd(x0, f, j_f, tol, max_its, root))