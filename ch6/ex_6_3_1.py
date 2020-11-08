import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    x_min = -5
    x_max = +5
    y_min = -5
    y_max = +5
    n = 15

    # Draw vector field
    x, y = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    u = x - y
    v = x**2 - 4

    plt.title('Exercise 6.3.1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.quiver(x, y, u, v)
    plt.show()