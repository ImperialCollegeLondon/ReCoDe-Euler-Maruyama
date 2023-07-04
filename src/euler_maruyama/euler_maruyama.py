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
    _t_0: float
        Initial time.

    _t_n: float
        Final time.

    _n_steps: int
        Number of time steps to discretise the time interval [t_0, t_n].

    _X_0: float
        Initial condition of the SDE.

    _drift: Coefficient
        Drift (mu) coefficient of the SDE.

    _diffusion: Coefficient
        Diffusion (sigma) coefficient of the SDE.

    _n_sim: int
        Number of simulated approximations.

    delta: float
        Length of the time step.

    t: np.ndarray
        Array containing the time steps values, shape: (n_steps+1,).

    Y: np.ndarray
        Array containing the approximated solution of the SDE, shape: shape(n_sim, n_steps+1).

    Methods
    -------
    compute_numerical_approximation
    """

    def __init__(
        self,
        t_0: float,
        t_n: float,
        n_steps: int,
        X_0: float,
        drift: Coefficient,
        diffusion: Coefficient,
        n_sim: int,
    ):

        self._t_0 = t_0
        self._t_n = t_n
        self._n_steps = n_steps

        self._X_0 = X_0

        self._drift = drift
        self._diffusion = diffusion

        self._n_sim = n_sim

        self.Y = None
        self._compute_discretisation()

    @property
    def n_sim(self):
        return self._n_sim

    @n_sim.setter
    def n_sim(self, value: int):
        """Change the number of simulations.

        Parameters
        ----------
        value: int
            Number of simulations.
        """
        if value > 0:
            self._n_sim = value
        else:
            raise ValueError("Number of simulations must be positive.")

    @property
    def n_steps(self):
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value: int):
        """Change the number of time steps attribute and recalculate the discretisation.

        Parameters
        ----------
        value: int
            Number of time steps.
        """
        if value > 0:
            self._n_steps = value
            self._compute_discretisation()
        else:
            raise ValueError("Number of steps must be positive.")

    def _compute_discretisation(self) -> None:
        """Calculate time steps and time delta."""
        self.t, self.delta = np.linspace(
            self._t_0, self._t_n, self._n_steps + 1, retstep=True
        )

    def _allocate_Y(self, dim: int) -> np.ndarray:
        """Allocate an array for the approximated solution.

        Parameters
        ----------
        dim: int
            Number of simulations, dimension 0 of Y.

        Returns
        -------
        Y: np.ndarray
            Array for the approximated solution.
        """
        Y = np.zeros((dim, self._n_steps + 1), dtype=float)
        Y[:, 0] = self._X_0 * np.ones(dim)
        return Y

    def _solve_numerical_approximation(self, dim: int) -> np.ndarray:
        """Solve the EM approximation for the given number of simulated trajectories.

        Parameters
        ----------
        dim: int
            The number of simulations, dimension 0 of Y.

        Returns
        -------
        Y: np.ndarray
            Array containing the approximated solution of the SDE, shape(n_sim, n_steps+1).
        """
        Y = self._allocate_Y(dim=dim)
        for n in range(self._n_steps + 1)[:-1]:
            tau_n = self.t[n]
            Y_n = Y[:, n]

            mu = self._drift.get_value(X=Y_n, t=tau_n)
            sigma = self._diffusion.get_value(X=Y_n, t=tau_n)

            dW = np.random.normal(loc=0, scale=np.sqrt(self.delta), size=dim)

            # Compute next step of the EM scheme
            Y[:, n + 1] = Y_n + mu * self.delta + sigma * dW

        return Y

    def compute_numerical_approximation(self) -> np.ndarray:
        """Compute the EM approximation for all simulated trajectories.

        Returns
        -------
        Y: np.ndarray
            Array containing the approximated solution of the SDE, shape(n_sim, n_steps+1).
        """
        self.Y = self._solve_numerical_approximation(dim=self._n_sim)
        return self.Y
