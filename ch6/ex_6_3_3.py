import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    x_min = -2
    x_max = +2
    y_min = -2
    y_max = +2
    n = 15

    # Draw vector field
    x, y = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    u = 1. + y - np.exp(-x)
    v = x**3 - y

    plt.title('Exercise 6.3.3')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.quiver(x, y, u, v)
    plt.show()