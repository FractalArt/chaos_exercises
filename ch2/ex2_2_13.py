from scipy.optimize import fsolve
from numpy import cosh, log
import matplotlib.pyplot as plt
import numpy as np


# The function whose zeros we want to find
def numeric(V):
    return V**2/32.2*log(cosh(32.2*116/V))-29300.0


# Determine the terminal solution from the analytic expansion
def analytic(t, s, g=32.2):
    """
    g is given in ft/s^2, t in s and s in ft.
    """
    delta = np.sqrt(t**2 - 4*log(2)*s/g)
    return (t-delta)/(2*log(2)/g), (t+delta)/(2*log(2)/g)


if __name__ == "__main__":
    # Take as initial guess the average velocity
    initial_guess = 253.
    print(f"Terminal velocity (from numerical solution)  : {fsolve(numeric, initial_guess)[0]} ft/s")
    print(f"Terminal velocity (from analytic expansion) 1: {analytic(116., 29300)[0]} ft/s")
    print(f"Terminal velocity (from analytic expansion) 2: {analytic(116., 29300)[1]} ft/s")
    # Determine a value for k
    k = 32.2*261.2/265.69**2
    print(f"k={k} pounds / ft")
    
    # Check the terminal velocity also graphically
    values = np.linspace(240., 270., 1000)
    evals = [numeric(val) for val in values]
    plt.plot(values, np.zeros(len(evals)), color="black")
    plt.plot(values, evals)
    plt.title("Determination of the Terminal Velocity")
    plt.xlabel("V [ft/s]")
    plt.ylabel("f(V)")
    plt.show()
