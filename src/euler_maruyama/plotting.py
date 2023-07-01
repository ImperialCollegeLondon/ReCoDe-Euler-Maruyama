import matplotlib.pyplot as plt
import numpy as np


def plot_approximation(Y: np.ndarray, t: np.ndarray, title: str) -> None:
    """Plot the numerical approximation obtained for the Stochastic Differential Equation.

    Parameters
    ----------
    Y: np.ndarray
        The approximated trajectories of the Euler-Maruyama method, shape = (number of simulations, number of steps).

    t: np.ndarray
        Array containing the time steps values, shape: (number of steps, ).

    title: str
        Title of the figure.
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(t, Y.T, alpha=0.3)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$Y_t$")

    ax.set_title(title)

    plt.show()
