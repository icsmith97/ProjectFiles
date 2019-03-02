import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(-2, 2, 0.25)
y = np.arange(-1, 3, 0.25)
x, y = np.meshgrid(x, y)

f = (1 - x)**2 + 100*(y - x**2)**2

surf = ax.plot_surface(x, y, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(0, 2500)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$y$', fontsize=15)
ax.set_ylabel('$Y$')
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-1, 0, 1, 2, 3])
ax.set_zticks([0, 500, 1000, 1500, 2000, 2500])

for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(14)
for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(14)
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(14)


plt.show()
fig.savefig("RosenbrockFunction.pdf", bbox_inches='tight')