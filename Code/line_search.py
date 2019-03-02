def backtracking(f, grad_fk, pk, xk):
    a = 1
    rho = 0.5
    c = 0.1

    while f(xk + (a * pk)) > (f(xk) + (c * a) * (grad_fk.transpose() * pk)):
        a = rho * a

    return a
