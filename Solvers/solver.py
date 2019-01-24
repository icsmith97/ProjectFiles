import numpy as np
import matplotlib.pyplot as plt         # Contains functions to plot graphs.
from matplotlib import rc               # Allows access to the LaTeX interpreter for formatting graphs.

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


# Finite Difference Approximation?

class Solver:
    def __init__(self, initial_guess, maximum_iterations, tolerance):
        self.x0 = initial_guess
        self.maxIts = maximum_iterations
        self.tol = tolerance
        self.its = 0
        self.time_taken = 0
        self.trace = []

    def __str__(self):
        output = "Initial Guess: {}\n".format(self.x0)
        output += "Approximated Root: {}\n".format(self.trace[-1])
        output += "Iterations Taken: {}\n".format(self.its)
        output += "Time Taken: {}\n".format(self.time_taken)
        return output

    def error_pair_plot(self, root):
        xs = self.trace
        n = len(xs)
        ys = []

        for i in range(n - 1):
            er_k = np.linalg.norm(xs[i] - root)
            er_kp1 = np.linalg.norm(xs[i + 1] - root)
            y = np.log(er_kp1) - np.log(er_k)
            ys.append(y)

        plot_x = np.arange(n - 1) + 1
        plot_y = np.array(ys)

        plt.figure(1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r'$\log e_k$', fontsize=14)
        plt.ylabel(r'$\log e_{k+1}$', fontsize=14)
        plt.grid(True)
        plt.title(r'Log Error Plot', fontsize=16)
        plt.plot(plot_x, plot_y, 'ro')
