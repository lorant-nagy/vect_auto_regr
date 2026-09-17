import numpy as np
import pytest

from vect_autoreg import VectAutoReg


def test_companion_matrix_arbitrary_order():
    d, p = 2, 4
    coeffs = [np.eye(d) * (0.1 / (i + 1)) for i in range(p)]
    f = VectAutoReg.stability_companion(coeffs)

    assert f.shape == (d * p, d * p)
    np.testing.assert_allclose(f[:d], np.hstack(coeffs))
    np.testing.assert_allclose(f[d:, :-d], np.eye(d * (p - 1)))
    np.testing.assert_allclose(f[d:, -d:], 0.0)


def test_one_innovation_per_step_variance_does_not_depend_on_order():
    n = 50_000
    variances = []
    for p in (1, 2, 5):
        coeffs = [np.zeros((1, 1)) for _ in range(p)]
        model = VectAutoReg(coeffs, rng=123)
        series = model(n + p)
        # Skip the initial p zeros; every generated value should be N(0,1).
        variances.append(np.var(series[0, p:]))

    for variance in variances:
        assert variance == pytest.approx(1.0, abs=0.03)


def test_returned_length_is_exactly_requested_length():
    model = VectAutoReg([np.array([[0.2]]), np.array([[0.1]])], rng=1)
    series = model(25)
    assert series.shape == (1, 25)
    assert len(model) == 25


def test_bias_and_covariance_are_used():
    model = VectAutoReg(
        [np.zeros((2, 2))],
        bias=[2.0, -1.0],
        noise_cov=np.zeros((2, 2)),
        rng=1,
    )
    series = model(4)
    # First point is X_0=0; subsequent points equal the bias exactly.
    np.testing.assert_allclose(series[:, 1:], np.array([[2.0] * 3, [-1.0] * 3]))


def test_stationary_initialization_for_ar1():
    a = 0.8
    sigma2 = 2.0
    model = VectAutoReg(
        [np.array([[a]])],
        bias=[1.5],
        noise_cov=[[sigma2]],
        rng=1234,
    )
    expected_mean = 1.5 / (1 - a)
    expected_var = sigma2 / (1 - a * a)

    np.testing.assert_allclose(model.stationary_mean(), [expected_mean])
    np.testing.assert_allclose(model.stationary_state_cov(), [[expected_var]])

    draws = np.array(
        [model.sample_stationary_initial_history()[0, 0] for _ in range(20_000)]
    )
    assert draws.mean() == pytest.approx(expected_mean, abs=0.08)
    assert draws.var() == pytest.approx(expected_var, abs=0.15)


def test_stationary_init_rejected_for_unstable_process():
    model = VectAutoReg([np.array([[1.1]])], rng=1)
    with pytest.raises(ValueError, match="stable VAR"):
        model(10, stationary_init=True)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        VectAutoReg([])
    with pytest.raises(ValueError):
        VectAutoReg([np.ones((2, 3))])
    with pytest.raises(ValueError):
        VectAutoReg([np.eye(2), np.eye(3)])
    with pytest.raises(ValueError):
        VectAutoReg([np.eye(2)], bias=[1.0])
    with pytest.raises(ValueError):
        VectAutoReg([np.eye(2)], init=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        VectAutoReg([np.eye(2)], noise_cov=[[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        VectAutoReg([np.eye(2)], noise_cov=[[1.0, 2.0], [2.0, 1.0]])


def test_explicit_init_and_stationary_init_are_mutually_exclusive():
    model = VectAutoReg([np.array([[0.2]])], init=[[0.0]], rng=1)
    with pytest.raises(ValueError, match="explicit init"):
        model(5, stationary_init=True)


def test_stationary_state_cov_satisfies_lyapunov_equation_for_var2():
    a1 = np.array([[0.25, 0.05], [0.0, 0.2]])
    a2 = np.array([[0.1, 0.0], [0.03, 0.1]])
    sigma = np.array([[1.0, 0.2], [0.2, 0.7]])
    model = VectAutoReg([a1, a2], noise_cov=sigma, rng=1)
    assert model.stable

    f = model.companion_matrix
    p_cov = model.stationary_state_cov()
    q = np.zeros_like(p_cov)
    q[:2, :2] = sigma
    np.testing.assert_allclose(p_cov, f @ p_cov @ f.T + q, atol=1e-10, rtol=1e-10)
