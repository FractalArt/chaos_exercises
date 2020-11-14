# https://stackoverflow.com/questions/13828800/how-to-make-a-quiver-plot-in-polar-coordinates
import matplotlib.pyplot as plt
import numpy as np

radii = np.linspace(0.1, 1.3, 10)
thetas = np.linspace(0., 2.*np.pi, 30)

theta, r = np.meshgrid(thetas, radii)

dr = r * (1. - r**2)
dt = 1. - np.cos(theta)

f = plt.figure()
ax = f.add_subplot(111, polar=True)
ax.quiver(theta, r, dr * np.cos(theta) - dt * np.sin(theta), dr * np.sin(theta) + dt * np.cos(theta))
plt.show()
