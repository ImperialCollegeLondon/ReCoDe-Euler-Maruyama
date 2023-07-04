import numpy as np
from joblib import Parallel, delayed

from .coefficients import Coefficient
from .euler_maruyama import EulerMaruyama


class ParallelEulerMaruyama(EulerMaruyama):
    """Class to perform the numerical solution of a Stochastic Differential Equation (SDE) through the Euler-Maruyama method
    using parallel computation.

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

    n_jobs: int
        Number of batches to compute in parallel.

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

    _n_jobs: int
        Number of batches to compute in parallel.

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
        n_jobs: int,
    ):
        super().__init__(
            t_0=t_0,
            t_n=t_n,
            n_steps=n_steps,
            X_0=X_0,
            drift=drift,
            diffusion=diffusion,
            n_sim=n_sim,
        )
        self._n_jobs = n_jobs

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        """Change the number of batches.

        Parameters
        ----------
        value: int
            Number of batches.
        """
        if value > 0:
            self._n_jobs = value
        else:
            raise ValueError("Number of batches must be positive.")

    def _num_sim_batch(self) -> list[int]:
        """Calculate the number of simulations within each batch for parallel computation.

        Returns
        -------
        batches: list[int]
            List containing the number of simulations to include in each batch.
        """
        batch_size = self._n_sim // self._n_jobs
        remainder = self._n_sim % self._n_jobs

        batches = [batch_size] * (self._n_jobs - remainder) + [
            batch_size + 1
        ] * remainder

        return batches

    def compute_numerical_approximation(self) -> np.ndarray:
        """Compute the EM approximation for all simulated trajectories using parallel computing.

        Returns
        -------
        Y: np.ndarray
            Array containing the approximated solution of the SDE, shape(n_sim, n_steps+1).
        """
        Y_dim_batch_list = self._num_sim_batch()

        Y = Parallel(n_jobs=self._n_jobs)(
            delayed(self._solve_numerical_approximation)(dim=Y_dim)
            for Y_dim in Y_dim_batch_list
        )

        self.Y = np.concatenate(Y, axis=0)

        return self.Y
