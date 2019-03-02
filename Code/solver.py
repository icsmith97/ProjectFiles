import numpy as np
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

np.set_printoptions(5)

class Solver:
    def __init__(self, initial_guess, maximum_iterations, tolerance):
        self.x0 = initial_guess
        self.maxIts = maximum_iterations
        self.tol = tolerance
        self.its = 0
        self.time_taken = 0
        self.trace = []
        self.n = len(initial_guess)

    def __str__(self):
        if self.its != self.maxIts:
            output = "Initial Guess: {}\n".format(self.x0)
            output += "Approximated Root: {}\n".format(((self.trace[-1]).reshape(1, self.n)))
            output += "Iterations Taken: {}\n".format(self.its)
            output += "Time Taken: {:.5f}\n".format(self.time_taken)
        else:
            output = "Initial Guess: {}\n".format(self.x0)
            output += "Failed to converge in {} iterations\n".format(self.maxIts)
            output += "Final iterate is: {}.".format(((self.trace[-1]).reshape(1, self.n)))
            output += "Time Taken: {:.5f}\n".format(self.time_taken)
        return output

    def error_pair_vectors(self, root):
        xs = self.trace
        n = len(xs)
        er_ks = []
        er_kp1s = []

        for i in range(n - 1):
            er_k = np.linalg.norm(xs[i] - root)
            er_kp1 = np.linalg.norm(xs[i + 1] - root)
            er_ks.append(np.log(er_k))
            er_kp1s.append(np.log(er_kp1))

        plot_x = np.array(er_ks)
        plot_y = np.array(er_kp1s)

        return plot_x, plot_y
