import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    x_min = -4
    x_max = +4
    y_min = -4
    y_max = +4
    n = 8

    # Draw vector field
    x, y = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    u = y**3 - 4. * x
    v = y**3 - y - 3.0 * x

    plt.title('Exercise 6.3.9')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.quiver(x, y, u, v)
    plt.show()