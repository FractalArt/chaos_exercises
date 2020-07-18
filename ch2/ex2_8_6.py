import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x + np.exp(-x)


def plot_gradient_field(ax, func, title=None):
    X = np.arange(-4, 4, 0.1)
    Y = np.arange(-4, 4, 0.1)
    U, V = np.meshgrid(np.ones(len(X)), func(X))
    ax.quiver(X, Y, U/3, V/3)

    ax.set_xlabel('x')
    ax.set_ylabel('x\'')
    if title:
        ax.set_title(title)

def euler_method(f, t0: float, x0: float, timestep: float, end: float, exact_solution=None):
    """
    Implementation of the euler method to numerically compute the solution
    to the differential equation

                x'=f(x)

    Parameters
    ----------
    f: function
        The implementation of the function `f` appearing in the differential
        equation.
    t0: float
        The initial time.
    x0: float
        The initial condition to the differential equation, i.e. the value
        of x(t=t0).
    timestep: float
        The timestep to employ for the numerical solution of the differential
        equation.
    end: float
        The maximal time step up to which to compute the the solution.
    exact_solution: function
        The exact solution. If the value is different from `None` the exact
        solution will
        be evaluated at each time step and the corresponding values will be
        returned in order
        to be able to check the convergence of the numerical solution.

    """
    if end < t0:
        raise ValueError("Initial time is larger than the end time!")

    # Store the time steps
    time_steps = [t0]
    # Store the value at each time step
    values = [x0]
    # Store the exact values of the solutions at each time step, if the exact
    # solution is provided
    if exact_solution:
        exact_values = [exact_solution(t0)]

    # Now start solving the differential equation numerically
    t = t0
    x = x0
    while t < end:
        t = t + timestep
        time_steps.append(t)
        x = x + f(x) * timestep
        values.append(x)
        if exact_solution:
            exact_values.append(exact_solution(t))

    return time_steps, values, None if not exact_solution else exact_values


def runge_kutta_method(f, t0: float, x0: float, timestep: float, end: float, exact_solution=None):
    """
    Implementation of the Runge-Kutta method to numerically compute the solution
    to the differential equation

                x'=f(x)

    Parameters
    ----------
    f: function
        The implementation of the function `f` appearing in the differential
        equation.
    t0: float
        The initial time.
    x0: float
        The initial condition to the differential equation, i.e. the value
        of x(t=t0).
    timestep: float
        The timestep to employ for the numerical solution of the differential
        equation.
    end: float
        The maximal time step up to which to compute the the solution.
    exact_solution: function
        The exact solution. If the value is different from `None` the exact
        solution will
        be evaluated at each time step and the corresponding values will be
        returned in order
        to be able to check the convergence of the numerical solution.

    """
    if end < t0:
        raise ValueError("Initial time is larger than the end time!")

    # Store the time steps
    time_steps = [t0]
    # Store the value at each time step
    values = [x0]
    # Store the exact values of the solutions at each time step, if the exact
    # solution is provided
    if exact_solution:
        exact_values = [exact_solution(t0)]

    # Now start solving the differential equation numerically
    t = t0
    x = x0
    while t < end:
        t = t + timestep
        time_steps.append(t)
        
        k1 = f(x) * timestep
        k2 = f(x + 0.5 * k1) * timestep
        k3 = f(x + 0.5 * k2) * timestep
        k4 = f(x + k3) * timestep

        x = x + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        values.append(x)
        
        if exact_solution:
            exact_values.append(exact_solution(t))

    return time_steps, values, None if not exact_solution else exact_values


if __name__ == "__main__":
    # a) Draw the gradient filed
    fig, ax = plt.subplots()
    plot_gradient_field(ax, f)
    plt.tight_layout()
    # plt.show()

    # b) Euler Method with stepsize 0.001
    time_steps, values, _ = euler_method(f, 0, 0,  0.001, 1)
    print(f"Euler result with step size 0.001: {values[-1]}")

    time_steps, values, _ = runge_kutta_method(f, 0, 0,  1, 1)
    print(f"Runge-Kutta result with step size 1: {values[-1]}")

    # This gives a weird result
    time_steps, values, _ = runge_kutta_method(f, 0, 0,  0.1, 1)
    print(f"Runge-Kutta result with step size 0.1: {values[-1]}")

    time_steps, values, _ = runge_kutta_method(f, 0, 0,  0.001, 1)
    print(f"Runge-Kutta result with step size 0.001: {values[-1]}")

    print(f(-1))

