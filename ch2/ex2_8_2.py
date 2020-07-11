import matplotlib.pyplot as plt
import numpy as np


def plot_gradient_field(ax, func, title=None):
    X = np.arange(-5, 5, 1)
    Y = np.arange(-5, 5, 1)
    U, V = np.meshgrid(np.ones(len(X)), func(X))
    ax.quiver(X, Y, U, V)

    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    if title:
        ax.set_title(title)


def f1(x):
    return x


def f2(x):
    return 1.0 - x**2


def f3(x):
    return 1. - 4. * x * (1. - x)


def f4(x):
    return np.sin(x)


if __name__ == "__main__":
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
    plot_gradient_field(axs[0][0], f1, r'$\dot{x}=x$')
    plot_gradient_field(axs[0][1], f2, r'$\dot{x}=1-x^2$')
    plot_gradient_field(axs[1][0], f3, r'$\dot{x}=1-4x(1-x)$')
    plot_gradient_field(axs[1][1], f4, r'$\dot{x}=sin(x)$')
    fig.tight_layout()
    plt.show()