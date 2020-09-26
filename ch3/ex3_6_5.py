#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graphical solution to the exercise 3.6.5.
"""
import matplotlib.pyplot as plt
import numpy as np

def f(u):
    """
    Implementation of the left-hand side of the reorganized dimensionless
    equilibrium equation:

        (1-1/u) \sqrt{1+u^2} = R
    """
    return (1. - 1. / u) * np.sqrt(1. + u**2) 

if __name__ == "__main__":

    # number of numerical samples to plot
    N_POINTS = 1000

    # array of u-vales
    u = np.linspace(-15, 15, N_POINTS)
    
    y = f(u)

    plt.plot(u, y, label=r'$y=(1-\frac{1}{u})\sqrt{1+u^2}$')
    plt.plot(u, np.array([2.0 for _ in range(N_POINTS)]), label=r'$y=R=2$')
    plt.plot(u, np.array([0.5 for _ in range(N_POINTS)]), label=r'$y=R=0.5$')
    plt.plot(u, np.array([4.0 for _ in range(N_POINTS)]), label=r'$y=R=4$')
    plt.ylim(0, 5)
    plt.legend()
    plt.show()




