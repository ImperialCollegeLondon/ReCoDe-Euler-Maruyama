import numpy as np

from euler_maruyama.coefficients import LinearDrift


def test_LinearDrift():

    a = 2.5
    linear_drift = LinearDrift(a=a)

    X_input = np.random.randn(20, 1)
    t_input = 10

    output = linear_drift.get_value(X=X_input, t=t_input)

    expected_output = np.ones((20, 1)) * t_input * a

    np.testing.assert_array_equal(output, expected_output)
