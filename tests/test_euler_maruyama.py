import numpy as np
import pytest

from euler_maruyama import ConstantDiffusion, EulerMaruyama, LinearDrift


@pytest.fixture(scope="function")
def em():

    linear_drift = LinearDrift(a=2)
    constant_diffusion = ConstantDiffusion(b=0.5)

    em = EulerMaruyama(
        t_0=0,
        t_n=2,
        n_steps=500,
        X_0=1.0,
        drift=linear_drift,
        diffusion=constant_diffusion,
        n_sim=1000,
    )

    return em


def test_discretisation(em):

    expected_t, expected_delta = np.linspace(
        em._t_0, em._t_n, em.n_steps + 1, retstep=True
    )
    np.testing.assert_array_equal(em.t, expected_t)
    assert em.delta == expected_delta


def test_n_sim(em):

    assert em.n_sim == 1000

    em.n_sim = 10

    assert em.n_sim == 10

    with pytest.raises(ValueError) as ex_info:
        em.n_sim = -10

    assert ex_info.value.args[0] == "Number of simulations must be positive."


def test_n_steps(em):

    assert em.n_steps == 500

    em.n_steps = 10

    assert em.n_steps == 10
    expected_t, expected_delta = np.linspace(em._t_0, em._t_n, 10 + 1, retstep=True)
    np.testing.assert_array_equal(em.t, expected_t)
    assert em.delta == expected_delta


def test_allocate_Y(em):

    Y = em._allocate_Y(dim=em.n_sim)

    assert Y.shape == (em.n_sim, em.n_steps + 1)

    expected_Y_0 = np.ones(em.n_sim) * em._X_0
    np.testing.assert_array_equal(Y[:, 0], expected_Y_0)


def test_numerical_approximation(em):

    Y = em.compute_numerical_approximation()

    assert Y.shape == (em.n_sim, em.n_steps + 1)

    expected_Y_0 = np.ones(em.n_sim) * em._X_0
    np.testing.assert_array_equal(Y[:, 0], expected_Y_0)

    expected_mean_Y_t = em._X_0 + em._t_n * 2
    mean_Y_t = np.mean(Y[:, -1])
    assert expected_mean_Y_t == pytest.approx(mean_Y_t, 0.1)

    expected_std_Y_t = 0.5 * np.sqrt(em._t_n)
    std_Y_t = np.std(Y[:, -1])
    assert expected_std_Y_t == pytest.approx(std_Y_t, 0.1)
