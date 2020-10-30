#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from phase_plane.runge_kutta import runge_kutta

def setup_cli(args):
    parser = argparse.ArgumentParser(description="Exercise 6.1")
    parser.add_argument('-e', '--exercise', type=int, choices=[1,3, 5], default=1, help="The sub-exercise of 6.1 to treat.")
    return parser.parse_args(args)

def f_1(v):
    """
    The non-linear function on the RHS of exercise 6.1.1.
    """
    x = v[0]
    y = v[1]
    return np.array([x-y, 1. - np.exp(x)])


def get_uv_1(x, y):
    """Get the velocity field for exercise 6.1.1."""
    return x-y, 1. - np.exp(x)


def f_3(v):
    """
    The non-linear function on the RHS of exercise 6.1.3.
    """
    x = v[0]
    y = v[1]
    return np.array([x*(x-y), y*(2*x-y)])

def f_5(v):
    """
    The non-linear function on the RHS of exercise 6.1.3.
    """
    x = v[0]
    y = v[1]
    return np.array([x*(2-x-y), x-y])

def get_uv_5(x,y):
    """Get the velocity field for exercise 6.1.5"""
    return x*(2-x-y), x-y

def get_uv_3(x, y):
    """Get the velocity field for exercise 6.1.3."""
    return x*(x-y), y*(2*x-y)

def exercise_6_1(f, uv, initial_conds, x_min=-1, x_max=1, y_min=-1, y_max=1, n=15, t_min=0, t_max=1, delta_t=0.01, index=1, quiver_options={}):
    # Start by plotting the vector field
    x,y  = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    u, v = uv(x, y)

    plt.title(f"Solution to exercise 6.1.{index}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.quiver(x, y, u, v, **quiver_options)

    # Compute some solutions
    for index, init in enumerate(initial_conds):
        x0 = np.array([init[0], init[1]])
        t, xy = runge_kutta(f, x0, t_min, t_max, delta_t)

        x = xy.T[0]
        y = xy.T[1]
        plt.plot(x, y, c=f"C{index}")

    plt.show()


if __name__ == "__main__":
    args = setup_cli(sys.argv[1:])

    print(f"Treating exercise 6.1.{args.exercise}")
    if args.exercise == 1:
        exercise_6_1(f_1, get_uv_1, [(-0.5,-0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5), (0.1, 0.1), (-0.1, -0.1), (-0.75, -1), (-0.5, -1)], t_max=2, index=args.exercise)
    elif args.exercise == 3:
        exercise_6_1(f_3, get_uv_3, [(-0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.1, 0.1), (-0.1, -0.1), (-0.75, -1), (-0.5, -1), (0.7,0.7)], 
                     x_min=-2,
                     x_max=2,
                     y_min=-2,
                     y_max=2,
                     delta_t=0.1,
                     t_max=1000, index=args.exercise)
    elif args.exercise == 5:
        exercise_6_1(f_5, get_uv_5, [(-0.5,-0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5), (0.1, 0.1), (-0.1, -0.1), (-0.75, -1), (-0.5, -1)], t_max=0.5,
                    x_min=-2,
                    x_max=2,
                    y_min=-2,
                    y_max=2,
                    index=args.exercise)
