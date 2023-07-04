import numpy as np
import pytest

from euler_maruyama import ConstantDiffusion, LinearDrift, ParallelEulerMaruyama


@pytest.fixture(scope="function")
def parallel_em():

    linear_drift = LinearDrift(a=2)
    constant_diffusion = ConstantDiffusion(b=0.5)

    em = ParallelEulerMaruyama(
        t_0=0,
        t_n=2,
        n_steps=500,
        X_0=1.0,
        drift=linear_drift,
        diffusion=constant_diffusion,
        n_sim=1000,
        n_jobs=4,
    )

    return em


def test_n_jobs(parallel_em):

    assert parallel_em.n_jobs == 4

    parallel_em.n_jobs = 2
    assert parallel_em.n_jobs == 2

    with pytest.raises(ValueError) as ex_info:
        parallel_em.n_jobs = -5

    assert ex_info.value.args[0] == "Number of batches must be positive."


def test_num_sim_batch(parallel_em):

    batches = parallel_em._num_sim_batch()

    assert batches == [250] * 4

    parallel_em.n_jobs = 3

    new_batches = parallel_em._num_sim_batch()

    assert new_batches == [333] * (3 - 1) + [334] * 1

    parallel_em.n_jobs = 7

    new_batches_2 = parallel_em._num_sim_batch()

    assert new_batches_2 == [142] * (7 - 6) + [143] * 6


def test_numerical_approximation(parallel_em):

    Y = parallel_em.compute_numerical_approximation()

    assert Y.shape == (parallel_em.n_sim, parallel_em.n_steps + 1)

    expected_Y_0 = np.ones(parallel_em.n_sim) * parallel_em._X_0
    np.testing.assert_array_equal(Y[:, 0], expected_Y_0)

    expected_mean_Y_t = parallel_em._X_0 + parallel_em._t_n * 2
    mean_Y_t = np.mean(Y[:, -1])
    assert expected_mean_Y_t == pytest.approx(mean_Y_t, 0.1)

    expected_std_Y_t = 0.5 * np.sqrt(parallel_em._t_n)
    std_Y_t = np.std(Y[:, -1])
    assert expected_std_Y_t == pytest.approx(std_Y_t, 0.1)
