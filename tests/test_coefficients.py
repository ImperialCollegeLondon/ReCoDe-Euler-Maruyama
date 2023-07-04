import numpy as np
import pytest

from euler_maruyama import (
    ConstantDiffusion,
    LinearDrift,
    MeanReversionDrift,
    MultiplicativeNoiseDiffusion,
)


@pytest.fixture(scope="module")
def X_input():
    X_input = np.random.randn(20, 1)
    return X_input


@pytest.fixture(scope="module")
def t_input():
    t_input = 10
    return t_input


def test_LinearDrift(X_input, t_input):

    a = 2.5
    linear_drift = LinearDrift(a=a)

    output = linear_drift.get_value(X=X_input, t=t_input)

    expected_output = np.ones_like(X_input) * t_input * a

    np.testing.assert_array_equal(output, expected_output)


def test_MeanReversionDrift(X_input, t_input):

    theta = 0.2
    mean = 1
    mean_reversion_drift = MeanReversionDrift(theta=theta, mean=mean)

    output = mean_reversion_drift.get_value(X=X_input, t=t_input)

    expected_output = theta * (mean - X_input)

    np.testing.assert_array_equal(output, expected_output)


def test_ConstantDiffusion(X_input, t_input):

    b = 0.1
    constant_diffusion = ConstantDiffusion(b=b)

    output = constant_diffusion.get_value(X=X_input, t=t_input)

    expected_output = np.ones_like(X_input) * b

    np.testing.assert_array_equal(output, expected_output)


def test_MultiplicativeNoiseDiffusion(X_input, t_input):

    b = 0.5
    multiplicative_noise_diffusion = MultiplicativeNoiseDiffusion(b=b)

    output = multiplicative_noise_diffusion.get_value(X=X_input, t=t_input)

    expected_output = X_input * b

    np.testing.assert_array_equal(output, expected_output)
