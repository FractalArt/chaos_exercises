"""
In this file we compare the performance of numerical methods to solve
differential equations describing first order systems. 
These include the Euler and improved-Euler as well as the Runge-Kutta 
methods.

We will apply them on the example of a simple first-order system described
by the differential equation

    x' = -x

which is simple enough to be able to derive the exact solution analytically
such that it can be used to assess the performance of the numerical method.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def setup_cli(args):
    """
    Setup the command line interface.
    """
    parser = argparse.ArgumentParser(description="Solve the differential equation x'=-x numerically.")
    parser.add_argument(
        "--t0",
        type=float,
        default=0.0,
        help="The starting time t0 at which the initial condition is defined. Default: t0=0.0",
    )
    parser.add_argument("--x0", type=float, default=1.0, help="The initial condition x(t=t0)=x0. Default: x0=1.0")
    parser.add_argument("--step", type=float, default=0.0001, help="The step size. Default: s=0.0001")
    parser.add_argument("--end", type=float, default=1.0, help="The end point in the time evolution. Default: end=1.0")
    return parser.parse_args()


def exact_solution(t, x0=1.0):
    """
    The implementation of the exact solution to the differential equation
    x' = -x
    with the initial condition x(t=0)=`x0` (default x0=1, as in the problem).
    """
    return x0 * np.exp(-t)


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

    return np.array(time_steps), np.array(values), None if not exact_solution else np.array(exact_values)


def improved_euler_method(f, t0: float, x0: float, timestep: float, end: float, exact_solution=None):
    """
    Implementation of the improved euler method to numerically compute the solution
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
        x_tilde = x + f(x) * timestep
        x = x + 0.5 * (f(x) + f(x_tilde)) * timestep
        values.append(x)
        if exact_solution:
            exact_values.append(exact_solution(t))

    return np.array(time_steps), np.array(values), None if not exact_solution else np.array(exact_values)


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

    return np.array(time_steps), np.array(values), None if not exact_solution else np.array(exact_values)


if __name__ == "__main__":
    # Imports
    import sys
    import logging

    # Setup the logger
    logging.basicConfig(level=logging.INFO)

    # Setup the command-line parser and extract the parameters
    args = setup_cli(sys.argv)
    x0 = args.x0
    t0 = args.t0
    stop = args.end
    step_size = args.step

    # Log the parameters that we employ
    logging.info(f"Using initial time      t0  = {t0}")
    logging.info(f"Using initial condition x0  = {x0}")
    logging.info(f"Using step size         s   = {step_size}")
    logging.info(f"Using end time          end = {stop}")

    # Compare the performances
    numerical_methods = {
        "Euler": euler_method,
        "Improved Euler": improved_euler_method,
        "Runge-Kutta": runge_kutta_method,
    }

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for label, f in numerical_methods.items():
        ts, vs, exact = f(lambda x: -x, t0, x0, step_size, stop, exact_solution)
        error = np.abs(vs - exact)
        ax1.plot(ts, vs, label=label.lower())
        ax2.plot(ts, error, label=label.lower())

    time_exact = np.linspace(t0, stop, 1000)
    vals_exact = exact_solution(time_exact)
    ax1.plot(time_exact, vals_exact, label="exact")

    # Cosmetics for the plot
    ax1.set_title(r"Solution to the first-order system $\dot{x}=-x$")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$|x_{\mathrm{num}}(t) - x_{\mathrm{exact}}(t)|$")
    ax1.legend(loc="best")
    # ax1.set_yscale('log')
    # ax1.set_xscale('log')
    fig1.tight_layout()

    ax2.set_title(r"Error of the solution to the first-order system $\dot{x}=-x$")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$|x_{\mathrm{num}}(t) - x_{\mathrm{exact}}(t)|$")
    ax2.legend(loc="best")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    fig2.tight_layout()
    plt.show()
    # fig1.savefig("compare_numerical_solvers.pdf")
    # fig2.savefig("compare_numerical_solvers_error.pdf")
