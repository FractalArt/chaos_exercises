import matplotlib.pyplot as plt
import numpy as np


def f(E, l, kappa=1):
    """
    The implementation of the function f in the differential equation

    E' = f(E, l, k) 

    with

    f(E, l, k) = k * E * l * (1 - E^2) / (1 + l*E^2) 
    """
    return kappa * E * l * (1. - E**2) / (1. + l * E**2)


def plot_for_lambda(l):
    """
    Plot the vector field as a function of the parameter l.
    """
    e = np.linspace(-2, 2, 1000)
    e_dot = f(e, l)
    plt.plot(e, e_dot, label=rf"$\lambda={l}$")


if __name__ == "__main__":
    # plot_for_lambda(-2.)
    # plot_for_lambda(-1.05)
    plot_for_lambda(-1.)
    plot_for_lambda(-0.5)
    plot_for_lambda(0.)
    plot_for_lambda(1.)
    plt.legend()
    plt.show()