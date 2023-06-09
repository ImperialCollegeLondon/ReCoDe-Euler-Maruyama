import matplotlib.pyplot as plt
import numpy as np

from .coefficients import Coefficient


class EulerMaruyama:
    """Class to perform the numerical solution of a Stochastic Differential Equation (SDE) through the Euler-Maruyama method.

    Considering a SDE of the form: dX_t = mu(X_t, t)dt + sigma(X_t, t)dW_t, the solution of this SDE over
    the time interval [t_0, t_n] can be approximated as follows:

                    Y_{n+1} = Y_n + mu(Y_n, tau_n)(tau_{n+1} -  tau_n) + sigma(Y_n, tau_n)(W_{tau_{n+1}} - W_{tau_n})

    with initial condition Y_0 = X_0 and where the time interval is discretised:

                    t_0 = tau_0 < tau_1 <  ... < tau_n  = t_n

    with Delta_t = tau_{n+1} - tau_n = (t_n - t_0) / n and DeltaW_n = (W_{tau_{n+1}} - W_{tau_n}) ~ N(0, Delta_t)
    because W_t is a Wiener process, so-called Brownian motion.

    Parameters
    ----------
    t_0: float
        Initial time.

    t_n: float
        Final time.

    n_steps: int
        Number of time steps to discretise the time interval [t_0, t_n].

    X_0: float
        Initial condition of the SDE.

    drift: Coefficient
        Drift (mu) coefficient of the SDE.

    diffusion: Coefficient
        Diffusion (sigma) coefficient of the SDE.

    n_sim: int
        Number of simulated approximations.

    Attributes
    ----------
    delta: float
        Length of the time step.

    steps: np.ndarray
        Array containing the time steps ordinals, shape: (n_steps+1,).

    t: np.ndarray
        Array containing the time steps values, shape: (n_steps+1,).

    Y: np.ndarray
        Array containing the approximated solution of the SDE, shape: shape(n_sim, n_steps).

    Methods
    -------
    compute_numerical_approximation
    plot_approximation
    """

    def __init__(self, t_0: float, t_n: float, n_steps: int, X_0: float, drift: Coefficient, diffusion: Coefficient, n_sim: int):

        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps

        self.X_0 = X_0

        self.drift = drift
        self.diffusion = diffusion

        self.n_sim = n_sim

        self.delta, self.steps, self.t = self._compute_discretisation()
        self.Y = self._allocate_Y()

    def _compute_discretisation(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Calculate time step length and number of steps array.

        Returns
        -------
        delta: float
            Length of the time step.

        steps: np.ndarray
            Array containing the time steps ordinals, shape: (n_steps+1,).

        t: np.ndarray
            Array containing the time steps values, shape: (n_steps+1,).
        """
        delta = (self.t_n - self.t_0) / self.n_steps
        steps = np.arange(0, self.n_steps + 1)
        t = self.t_0 + steps * delta
        return delta, steps, t

    def _allocate_Y(self):
        """Allocate an array for the approximated solution.

        Returns
        -------
        np.ndarray
            Array containing the approximated solution of the SDE, shape (n_sim, n_steps+1).
        """
        Y = np.zeros((self.n_sim, self.n_steps+1), dtype=float)
        Y[:, 0] = self.X_0 * np.ones(self.n_sim)
        return Y

    def compute_numerical_approximation(self):
        """Compute the EM approximation.

        Returns
        -------
        Y: np.ndarray
            Array containing the approximated solution of the SDE, shape(n_sim, n_steps).
        """
        self.Y = self._allocate_Y()  # Y must be reset at each execution
        for n in self.steps[:-1]:
            tau_n = self.t[n]
            Y_n = self.Y[:, n]

            mu = self.drift.get_value(X=Y_n, t=tau_n)
            sigma = self.diffusion.get_value(X=Y_n, t=tau_n)

            dW = np.random.normal(loc=0, scale=np.sqrt(self.delta), size=self.n_sim)

            # Compute next step of the EM scheme
            self.Y[:, n + 1] = Y_n + mu * self.delta + sigma * dW

        return self.Y

    def plot_approximation(self, title: str):
        """Plot the numerical approximation obtained for the SDE.

        Parameters
        ----------
        title: str
            Title of the figure
        """

        fig, ax = plt.subplots(figsize=(10, 7))

        y_T = self.Y.T
        ax.plot(self.t, y_T, alpha=0.3)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$Y_t$")

        ax.set_title(title)

        plt.show()


#TODO: Add docstring
class CustomEulerMaruyama(EulerMaruyama):
    """

    Methods
    -------
    change_n_sim
    change_n_steps
    """

    def __init__(self, t_0: float, t_n: float, n_steps: int, X_0: float, drift: Coefficient, diffusion: Coefficient, n_sim: int):
        super().__init__(t_0=t_0, t_n=t_n, n_steps=n_steps, X_0=X_0, drift=drift, diffusion=diffusion, n_sim=n_sim)

    def change_n_sim(self, new_n_sim: int):
        """Change the number of simulated approximations.

        Parameters
        ----------
        new_n_sim: int
            The new number of simulated approximations of the EM method.
        """
        self.n_sim = new_n_sim

    def change_n_steps(self, new_n_steps: int):
        """Change the number of steps attributes and recalculate the discretisation.

        Parameters
        ----------
        new_n_steps: int
            The new number of steps of the EM method.
        """
        self.n_steps = new_n_steps
        self.delta, self.steps, self.t = self._compute_discretisation()
