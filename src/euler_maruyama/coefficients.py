from abc import ABC, abstractmethod

import numpy as np


class Coefficient(ABC):
    """Abstract class to define the internal structure of drift and diffusion coefficients.

    Methods
    -------
    get_value
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError


class LinearDrift(Coefficient):
    """Implement a linear drift of the form:

                        mu(X_t, t) = a*t

    where a is a real value parameter.

    Parameters
    ---------
    a: float
        The linear coefficient of drift

    Methods
    -------
    get_value
    """

    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        return np.ones_like(X) * self.a*t


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
        return self.b * X
