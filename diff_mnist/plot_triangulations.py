
# Source code: 
# https://www.scaler.com/topics/matplotlib/matplotlib-triangulation/



# Colour map



# Example 1: Sphere

# This code is written in python
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams["figure.figsize"]=(10,10)
ax = plt.figure().gca(projection='3d')

a = np.linspace(0, 2 * np.pi,25)
b = np.linspace(0, np.pi, 25)

[X, Y] = np.meshgrid(a, b)

x = np.outer(np.cos(a), np.sin(b))
y = np.outer(np.sin(a), np.sin(b))
z = np.outer(np.ones_like(a), np.cos(b))


ax.plot_trisurf(
        x.flatten(), y.flatten(), z.flatten(),
        triangles=Delaunay(np.array([X.flatten(), Y.flatten()]).T).simplices,
        cmap=cm.RdPu
    )

# remove axes
ax.set_axis_off()
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# change the angle of view
# make the view
ax.view_init(30, 60)

# no background
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))#

plt.savefig('figures/sphere.pdf', bbox_inches='tight', pad_inches=0, dpi=600)

plt.show()




# ----------------------------------------------------------------------
# Example 2: Möblius strip


# This code is written in python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ax = plt.figure().gca(projection='3d')
plt.rcParams["figure.figsize"]=(10,10)

a = np.linspace(0, 2.0 * np.pi,30)
b = np.linspace(-0.5, 0.5, 5)
a, b = np.meshgrid(a, b)
a, b = a.flatten(), b.flatten()

# This is the Mobius mapping, taking a, b pair and returning an x, y, z
x = (1 + 0.5 * b * np.cos(a / 2.0)) * np.cos(a)
y = (1 + 0.5 * b * np.cos(a / 2.0)) * np.sin(a)
z = 0.5 * b * np.sin(a / 2.0)

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(a, b)


ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Greens)
ax.set_zlim(-1, 1)

# remove axes
ax.set_axis_off()
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


# # change the angle of view
# # make the view
# ax.view_init(30, 60)

plt.savefig('figures/mobius.pdf', bbox_inches='tight', pad_inches=0, dpi=1200)

plt.show()