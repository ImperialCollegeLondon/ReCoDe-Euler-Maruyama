from abc import ABC, abstractmethod

import numpy as np

#TODO: Add docsting
class Coefficient(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_value(self, X: np.ndarray, t: float) -> float:
        raise NotImplementedError


class LinearDrift(Coefficient):

    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def get_value(self, X: np.ndarray, t: float) -> float:
        return np.ones_like(X) * self.a*t


class ConstantDiffusion(Coefficient):

    def __init__(self, b: float):
        super().__init__()
        self.b = b

    def get_value(self, X: np.ndarray, t: float) -> float:
        return np.ones_like(X) * self.b
