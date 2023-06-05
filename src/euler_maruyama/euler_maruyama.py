import numpy as np


class EulerMaruyama:
    """Class to perform the numerical solution of a Stochastic Differential Equation (SDE) through the Euler-Maruyama method.

    Considering a SDE of the form: dX_t = mu(X_t, t)dt + sigma(X_t, t)dW_t, the solution of this SDE over
    the time interval [t_0, t_n] can be approximated as follows:

                    Y_{n+1} = Y_n + mu(Y_n, tau_n)(tau_{n+1} -  tau_n) + sigma(Y_n, tau_n)(W_{tau_{n+1}} - W_{tau_n})

    with initial condition Y_0 = X_0 and where the time interval is discretised:

                    t_0 = tau_0 < tau_1 <  ... < tau_n  = t_n

    with Delta_t = tau_{n+1} - tau_n = (t_n - t_0) / n and DeltaW_n = (W_{tau_{n+1}} - W_{tau_n}) ~ N(0, Delta_t)
    because W_t is a Wiener process, so-called Brownian motion.
    """

    def __init__(self, t_0: float, t_n: float, n: int):

        self.t_0 = t_0
        self.t_n = t_n
        self.n = n

        self.delta, self.steps = self.compute_discretisation()

    def compute_discretisation(self) -> tuple[float, np.ndarray]:
        delta = (self.t_n - self.t_0) / self.n
        steps = np.arange(0, self.n + 1)
        return delta, steps
