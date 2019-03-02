from Code import line_search
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 5*x**4 - 2*x


def fp(x):
    return 20*x**3 - 2


def fpp(x):
    return 60*x**2


def gradient_descent(x):
    f_ls = lambda y : np.array(f(y))
    fp_ls = lambda y : np.array(fp(y))

    x_ls = np.array(x)
    grad_fk = fp_ls(x)
    pk = -grad_fk
    alpha = line_search.backtracking(f_ls, grad_fk, pk, x_ls)
    return x - alpha*f(x)

def newtons_method(x):
    return x - f(x)/fp(x)


def halleys_method(x):
    return x - 2*f(x)*fp(x) / (2 * fp(x)**2 - f(x) * fpp(x))


def error_vectors(trace, root):
    n = len(trace)
    er_ks = []
    er_kp1s = []

    for i in range(n - 1):
        er_k = abs(trace[i] - root)
        er_kp1 = abs(trace[i + 1] - root)
        er_ks.append(np.log(er_k))
        er_kp1s.append(np.log(er_kp1))

    plot_x = np.array(er_ks)
    plot_y = np.array(er_kp1s)

    return plot_x, plot_y


def test_methods(x0, tol, max_its, root):
    functions = ["Gradient Descent", "Newton's Method", "Halley's Method"]

    heading = "Initial Guess: {}\n".format(x0)
    heading += "Tolerance: {}\n".format(tol)
    heading += "Maximum Number of Iterations: {}\n\n".format(max_its)
    print(heading)

    traces = []

    for function_switch in range(3):
        xk = x0
        its = 0

        trace = [xk]

        while abs(f(xk)) > tol and its < max_its:
            if function_switch == 0:
                xkp1 = gradient_descent(xk)
            elif function_switch == 1:
                xkp1 = newtons_method(xk)
            else:
                xkp1 = halleys_method(xk)

            xk = xkp1
            trace.append(xk)
            its += 1

        traces.append(trace)
        output = "Method Used: {}\n".format(functions[function_switch])
        output += "Approximated Root: {}\n".format(xk)
        output += "Iterations Taken: {}\n".format(its)
        print(output)

    gd_trace = traces[0]
    nm_trace = traces[1]
    hm_trace = traces[2]

    gd_x, gd_y = error_vectors(gd_trace, root)
    nm_x, nm_y = error_vectors(nm_trace, root)
    hm_x, hm_y = error_vectors(hm_trace, root)

    g = plt.figure(1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$\log e_k$', fontsize=16)
    plt.ylabel(r'$\log e_{k+1}$', fontsize=16)
    plt.grid(True)

    plt.plot(np.unique(gd_x), np.poly1d(np.polyfit(gd_x, gd_y, 1))(np.unique(gd_x)), 'r')
    plt.plot(np.unique(nm_x), np.poly1d(np.polyfit(nm_x, nm_y, 1))(np.unique(nm_x)), 'b')
    plt.plot(np.unique(hm_x), np.poly1d(np.polyfit(hm_x, hm_y, 1))(np.unique(hm_x)), 'g')

    plt.plot(gd_x, gd_y, 'rx')
    plt.plot(nm_x, nm_y, 'bx')
    plt.plot(hm_x, hm_y, 'gx')

    plt.legend(["Gradient Descent", "Newton's Method", "Halley's Method"], fontsize=16)
    plt.show()

    slope_gd = np.polyfit(gd_x, gd_y, 1)[0]
    slope_nm = np.polyfit(nm_x, nm_y, 1)[0]
    slope_hm = np.polyfit(hm_x, hm_y, 1)[0]

    print("Convergence Rate (GD): {:.3f}".format(slope_gd))
    print("Convergence Rate (NM): {:.3f}".format(slope_nm))
    print("Convergence Rate (Halley): {:.3f}".format(slope_hm))

    g.savefig("OneDimConvergence.pdf", bbox_inches='tight')

x0 = 1.4
max_its = 100000
tol = 1e-12
root = 0.7368062997280773

test_methods(x0, tol, max_its, root)


