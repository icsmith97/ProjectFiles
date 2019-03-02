import matplotlib.pyplot as plt
import numpy as np

def graph(x):
    return 5*x**4 - 2*x


x = np.linspace(-2, 2, 200)
func = np.vectorize(graph)
y = func(x)

f = plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$f(x)$', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.axis([-1, 1.5, -1, 1.5])
plt.grid()
plt.axhline(0, linewidth=1, color='k')
plt.axvline(0, linewidth=1, color='k')
plt.plot([0,0.737], [0, 0], 'go', markersize=8)
plt.plot(x, y, 'r')
plt.show()

f.savefig("OneDimPlot.pdf", bbox_inches='tight')