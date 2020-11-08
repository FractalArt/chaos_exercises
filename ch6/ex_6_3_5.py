import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    x_min = -10
    x_max = +10
    y_min = -10
    y_max = +10
    n = 40

    # Draw vector field
    x, y = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    u = np.sin(y)
    v = np.cos(x)

    plt.title('Exercise 6.3.5')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.quiver(x, y, u, v)
    plt.show()