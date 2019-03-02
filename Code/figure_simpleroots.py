import matplotlib.pyplot as plt
import numpy as np

def graph(x):
    return (x - 0.1) * (x - 0.3)**2 * (x - 0.7)**3


x = np.linspace(0, 1, 200)
func = np.vectorize(graph)
y = func(x)

f = plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$y(x)$', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.text(0.65, -0.000025, "Non-Simple Root", fontsize=13)
plt.text(0.2, 0.00001, "Non-Simple Root", fontsize=13)
plt.text(-0.175, 0.00001, "Simple Root", fontsize=13)
plt.axis([-0.2, 1, -0.0002, 0.0002])
plt.axhline(0, linewidth=1, color='b')
plt.plot(x, y, 'r')
plt.plot([0.1, 0.3, 0.7], [0, 0, 0], 'go')
plt.show()

f.savefig("SimpleRootFinding.pdf", bbox_inches='tight')