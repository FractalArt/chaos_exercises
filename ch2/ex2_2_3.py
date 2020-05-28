"""
This script plots the analytic solution to the differential equation in 
exercise 2.2.3 of the book `Nonlinear Dynamics and Chaos` by Steven H. Strogatz. 
"""
import matplotlib.pyplot as plt
import numpy as np


def get_velocities(times, x0: float):
    # Fix the integration constant using the initial condition
    c = 1.0 / (x0 ** 2) - 1.0
    if x0 < 0.0:
        return -1.0 / np.sqrt(1.0 + c * np.exp(-2.0 * times))
    else:
        return +1.0 / np.sqrt(1.0 + c * np.exp(-2.0 * times))


if __name__ == "__main__":
    # The set of initial conditions that we use.
    initial_conditions = [2.0, 0.7, 0.5, 0.3, -0.3, -0.5, -0.7, -2.0]

    # The time steps we use
    time_steps = np.linspace(0, 3, 100)

    fig, ax = plt.subplots()

    plt.title(r"$\dot{x}=x-x^3$")
    plt.xlabel("t")
    plt.ylabel("x")
    for counter, ic in enumerate(initial_conditions):
        velocities = get_velocities(time_steps, ic)
        ax.plot(time_steps, velocities, color=f"C{counter}")

    # Add the fixed points
    plt.plot(time_steps, np.zeros(len(time_steps)), color="gray", linestyle="-.")
    plt.plot(time_steps, np.ones(len(time_steps)), color="gray", linestyle="--")
    plt.plot(time_steps, -np.ones(len(time_steps)), color="gray", linestyle="--")

    # Draw the plot
    # plt.savefig('ex_2_2_3.pdf')
    plt.show()
