from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)

f = (1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 \
    + (2.625 - x + x*(y**3))**2

surf = ax.plot_surface(x, y, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(0, 400000)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$y$', fontsize=15)
ax.set_ylabel('$Y$')
ax.set_xticks([-5, -2.5, 0, 2.5, 5])
ax.set_yticks([-5, -2.5, 0, 2.5, 5])
ax.set_zticks([0, 100000, 200000, 300000, 400000])

for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(14)
for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(14)
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(14)


plt.show()
fig.savefig("BealesFunction.pdf", bbox_inches='tight')