import matplotlib.pyplot as plt
import numpy as np

def power_negative_3_solution(t: float, x0: float):
    """
    This function implements the solution to the differential equation

    x' = -x^3

    with initial condition x(t=0) = x0.
    """
    return np.sqrt(1.0 / (2.0 * t + 1.0 / x0**2))

def power_negative_1_solution(t: float, x0: float):
    """
    This function implements the solution to the differential equation

    x' = -x

    with initial condition x(t=0) = x0.
    """
    return x0 * np.exp(-t)


if __name__ == "__main__":
    # Get the time points for the x-axis
    ts = np.linspace(0.0, 10.0, 1000)

    # Get the solution of the differential equation x'=-x^3 evaluated at these points
    x_pm3 = power_negative_3_solution(ts, x0=10.)
    
    # Get the solution to the differential equation x'=-x evaluated at these points
    x_pm1 = power_negative_1_solution(ts, x0=10.)

    # Plot the solutions for comparison
    plt.plot(ts, x_pm3, label=r'$\dot{x}=-x^3$')
    plt.plot(ts, x_pm1, label=r'$\dot{x}=-x$')
    
    # Cosmetics
    plt.title(r"Comparisons of the solutions to the DE's $\dot{x}=-x^3$ and $\dot{x}=-x$")
    plt.xlabel("t (a.u.)")
    plt.ylabel("x (a.u.)")
    plt.legend()
    plt.show()