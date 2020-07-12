"""
This file contains the solution to the problem
2.8.5 of the book `Nonlinear Dynamics and Chaos` by Steven H. Strogatz.
"""
import numpy as np
import matplotlib.pyplot as plt


def exact_solution(t, x0=1.0):
    """
    The implementation of the exact solution to the differential equation
    x' = -x
    with the initial condition x(t=0)=`x0` (default x0=1, as in the problem).
    """
    return x0 * np.exp(-t)


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

    x0 = 1
    t0 = 0
    stop = 1.0

    # Part a) of the exercise
    print(f"\nPart a):")
    print(f"The exact value of x(t) is 1/e, or approximately 1/e = {np.exp(-1)}")

    # Part b) of the exercise
    print(f"\nPart b):")
    time_steps, values, exact_values = runge_kutta_method(lambda x: -x, t0, x0, 1.0, stop, exact_solution)
    print(f"time steps  : {time_steps}")
    print(f"values      : {values}")
    print(f"exact values: {exact_values}")

    step_size_solutions = {}
    for n in [1, 2, 3, 4]:
        time_steps, values, exact_values = runge_kutta_method(lambda x: -x, t0, x0, 10**(-n), stop, exact_solution)
        step_size_solutions[n] = {"time_steps": time_steps, "values": values, "exact": exact_values}
        
    # Plot the different solutions and compare
    # Exact
    exact_time = np.linspace(0, 1, 100)
    exact = exact_solution(exact_time)
    plt.plot(exact_time, exact, label='exact')
    for n in [1, 2, 3, 4]:
        plt.plot(step_size_solutions[n]['time_steps'], step_size_solutions[n]['values'], label=f"n={n}")
    
    plt.title(r'Numerical vs. exact solution for different time steps ($10^{-n}$)')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x(t)$')
    plt.legend()
    plt.show()

    # Part a) of the exercise
    print(f"\nPart c):")
    print(f"Plotting error vs. step size.")
    errors = []
    for n in [1, 2, 3, 4]:
        error = np.abs(step_size_solutions[n]['values'][-1] - step_size_solutions[n]['exact'][-1])
        errors.append(error)

    plt.plot([1, 2, 3, 4], errors)
    plt.title(r'Error vs step size $10^{-n}$')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$|\hat{x}(1)-x(1)|$')
    plt.tight_layout()
    plt.show()
    # plt.savefig('error_vs_stepsize.pdf')

    # Plot ln(E) vs ln(t)
    print("Plotting error evolution with time")    
    for n in [1, 2, 3, 4]:
        t = step_size_solutions[n]['time_steps']
        values = np.array(step_size_solutions[n]['values'])
        exact = np.array(step_size_solutions[n]['exact'])
        errors = np.abs(values-exact)
        plt.plot(np.log(t), np.log(errors), label=f"n={n}")

    plt.title(r'Error evolution with time for different step sizes ($10^{-n}$).')
    plt.xlabel(r'$\ln{(t)}$')
    plt.ylabel(r'$\ln{(|\hat{x}(t)-x(t)|)}$')
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.show()
    # plt.savefig('error_evolution_in_time_log_scale.pdf')
