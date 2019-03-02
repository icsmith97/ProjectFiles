import matplotlib.pyplot as plt
import numpy as np

def graph(x):
    return (x - 1)**2 - 0.25

x = np.linspace(0, 10, 200)
func = np.vectorize(graph)
y = func(x)

def tangent_line(x):
    return 1.5 * (x - 1.75) + 0.3125

func2 = np.vectorize(tangent_line)

f = plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$f(x)$', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.axis([0.75, 2, -0.3, 0.5])
plt.plot(x, y, 'r')
plt.plot(x, func2(x), linestyle='--', color='k')
plt.text(1.58, -0.05, r'$x_1$', fontsize=20)
plt.text(1.78, -0.05, r'$x_0$', fontsize=20)
plt.plot([1.542, 1.75], [0, 0], 'go')
plt.legend([r'$f(x)$', r'$f\textquotesingle(x_0)$'], fontsize=20)
plt.axhline(0, linewidth=1, color='b')
plt.axvline(1.75, linestyle=':', color='g')
plt.axvline(1.542, linestyle=':', color='g')
plt.show()

f.savefig("NewtonIteration.pdf", bbox_inches='tight')