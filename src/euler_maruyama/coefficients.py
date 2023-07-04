from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Coefficient(ABC):
    """Abstract class to define the internal structure of the drift and diffusion coefficients.

    Methods
    -------
    get_value
    plot_X_sample
    plot_t_sample
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def plot_X_sample(self) -> None:
        """Plot the coefficient value for 100 samples of X between 0 and 10."""
        result = np.zeros(100)
        x_array = np.linspace(0, 10, 100)
        for i, x in enumerate(x_array):
            result[i] = self.get_value(X=np.array(x), t=0)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(x_array, result)

        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"Coefficient value")

        plt.show()

    def plot_t_sample(self) -> None:
        """Plot the coefficient value for 100 samples of t between 0 and 1."""
        result = np.zeros(100)
        t_array = np.linspace(0, 1, 100)
        for i, t in enumerate(t_array):
            result[i] = self.get_value(X=np.array(1.0), t=t)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(t_array, result)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"Coefficient value")

        plt.show()


class LinearDrift(Coefficient):
    """Implement a linear drift of the form:

                        mu(X_t, t) = a*t

    where a is a real value parameter.

    Parameters
    ---------
    a: float
        The linear coefficient of drift.

    Methods
    -------
    get_value
    """

    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        """Compute the linear drift value as mu(X_t, t) = a*t.

        Parameters
        ----------
        X: np.ndarray
            The X_t values, shape = (number of simulations, )

        t: flotat
            The time value.

        Returns
        -------
        np.ndarray
            The linear drift coefficient values, shape = (number of simulations, )
        """
        return np.ones_like(X) * self.a * t


class MeanReversionDrift(Coefficient):
    """Implement a mean-reversion drift of the form:

                        mu(X_t, t) = theta*(mean - X_t)

    where theta and mean are real value parameters.

    Parameters
    ---------
    theta: float
        The speed of reversion.

    mean: float
        The equilibrium value.

    Methods
    -------
    get_value
    """

    def __init__(self, theta: float, mean: float):
        super().__init__()
        self.theta = theta
        self.mean = mean

    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        """Compute the mean-reversion drift value as mu(X_t, t) = theta*(mean - X_t).

        Parameters
        ----------
        X: np.ndarray
            The X_t values, shape = (number of simulations, )

        t: flotat
            The time value.

        Returns
        -------
        np.ndarray
            The mean-reversion drift coefficient values, shape = (number of simulations, )
        """
        return self.theta * (self.mean - X)


class ConstantDiffusion(Coefficient):
    """Implement a constant diffusion of the form:

                        sigma(X_t, t) = b

    where b is a real value parameter.

    Parameters
    ---------
    b: float
        The constant diffusion value.

    Methods
    -------
    get_value
    """

    def __init__(self, b: float):
        super().__init__()
        self.b = b

    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        """Compute the constant diffusion value as sigma(X_t, t) = b.

        Parameters
        ----------
        X: np.ndarray
            The X_t values, shape = (number of simulations, )

        t: flotat
            The time value.

        Returns
        -------
        np.ndarray
            The constant diffusion coefficient values, shape = (number of simulations, )
        """
        return np.ones_like(X) * self.b


class MultiplicativeNoiseDiffusion(Coefficient):
    """Implement a multiplicative noise-like diffusion of the form:

                        sigma(X_t, t) = b*X_t

    where b is a real value parameter.

    Parameters
    ---------
    b: float
        The amplitude of the multiplicative noise.

    Methods
    -------
    get_value
    """

    def __init__(self, b: float):
        super().__init__()
        self.b = b

    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        """Compute the multiplicative noise-like diffusion value as sigma(X_t, t) = b*X_t.

        Parameters
        ----------
        X: np.ndarray
            The X_t values, shape = (number of simulations, )

        t: flotat
            The time value.

        Returns
        -------
        np.ndarray
            The multiplicative noise-like diffusion coefficient values, shape = (number of simulations, )
        """
        return self.b * X
