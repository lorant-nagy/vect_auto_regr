"""Vector autoregression simulation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


class VectAutoReg:
    r"""Gaussian vector autoregression (VAR) simulator.

    The model is

    .. math::

        X_t = c + \sum_{j=1}^p A_j X_{t-j} + \varepsilon_t,
        \qquad \varepsilon_t \stackrel{iid}{\sim} N(0, \Sigma).

    Parameters
    ----------
    coeffs:
        Ordered coefficient matrices ``[A1, ..., Ap]``. Every matrix must be
        square and have the same shape.
    bias:
        Optional intercept vector ``c``. Defaults to zero.
    init:
        Optional chronological initial history with shape ``(p, d)``:
        ``[X_{-p+1}, ..., X_0]``. If omitted, zeros are used unless
        ``stationary_init=True`` is requested when simulating.
    noise_cov:
        Innovation covariance ``Sigma``. Defaults to the ``d x d`` identity.
        It must be symmetric positive semidefinite.
    rng:
        Optional NumPy ``Generator`` or integer seed.
    """

    def __init__(
        self,
        coeffs: Sequence[ArrayLike],
        bias: Optional[ArrayLike] = None,
        init: Optional[ArrayLike] = None,
        noise_cov: Optional[ArrayLike] = None,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        self.coeffs = self._validate_coeffs(coeffs)
        self.order = len(self.coeffs)
        self.dim = self.coeffs[0].shape[0]

        self.bias = self._validate_vector(bias, "bias", default=np.zeros(self.dim))
        self.noise_cov = self._validate_noise_cov(noise_cov)

        if init is None:
            self.init = np.zeros((self.order, self.dim), dtype=float)
            self._explicit_init = False
        else:
            self.init = self._validate_init(init)
            self._explicit_init = True

        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        self.series: Optional[np.ndarray] = None
        self.length: Optional[int] = None
        self.companion: Optional[np.ndarray] = None

    def __call__(self, length: int, *, stationary_init: bool = False) -> np.ndarray:
        """Simulate and return exactly ``length`` observations.

        The returned array has shape ``(d, length)`` to preserve the original
        package's orientation. The first ``p`` observations are the initial
        history. Therefore ``length`` must be at least the VAR order ``p``.

        If ``stationary_init=True``, the initial history is freshly sampled
        from the unique stationary Gaussian law. This requires a stable VAR
        and cannot be combined with a user-supplied ``init``.
        """
        if not isinstance(length, (int, np.integer)) or isinstance(length, bool):
            raise TypeError("length must be an integer")
        if length < self.order:
            raise ValueError(
                f"length must be at least the VAR order ({self.order}); got {length}"
            )
        if stationary_init and self._explicit_init:
            raise ValueError("stationary_init=True cannot be combined with an explicit init")

        if stationary_init:
            history = self.sample_stationary_initial_history()
        else:
            history = self.init.copy()

        values = [row.copy() for row in history]
        while len(values) < length:
            values.append(self._next_from_history(values))

        self.series = np.asarray(values, dtype=float)
        self.length = length
        return self.series.T.copy()

    def __len__(self) -> int:
        return 0 if self.length is None else self.length

    def _next_from_history(self, history: Sequence[np.ndarray]) -> np.ndarray:
        # Exactly one innovation vector is added per time step.
        next_vec = self.bias.copy()
        for lag, coeff in enumerate(self.coeffs, start=1):
            next_vec += coeff @ history[-lag]
        next_vec += self.rng.multivariate_normal(np.zeros(self.dim), self.noise_cov)
        return next_vec

    def _next(self) -> np.ndarray:
        """Generate one observation from the currently stored history."""
        if self.series is None:
            raise RuntimeError("No current series. Call the model first.")
        history = [row for row in self.series]
        return self._next_from_history(history)

    def _generate_companion(self) -> np.ndarray:
        """Backward-compatible wrapper returning the companion matrix."""
        self.companion = self.stability_companion(self.coeffs)
        return self.companion.copy()

    def _compute_stability(self) -> bool:
        """Backward-compatible stability check for a generated companion matrix."""
        if self.companion is None:
            raise RuntimeError("Companion matrix not generated. Call _generate_companion() first.")
        return self.is_stable(self.companion)

    @property
    def companion_matrix(self) -> np.ndarray:
        """Return the standard ``dp x dp`` VAR companion matrix."""
        if self.companion is None:
            self.companion = self.stability_companion(self.coeffs)
        return self.companion.copy()

    @property
    def stable(self) -> bool:
        """Whether every eigenvalue of the companion matrix has modulus < 1."""
        return self.is_stable(self.companion_matrix)

    @staticmethod
    def stability_companion(coeffs: Sequence[ArrayLike]) -> np.ndarray:
        """Construct the standard companion matrix for arbitrary order ``p >= 1``."""
        matrices = VectAutoReg._validate_coeffs(coeffs)
        p = len(matrices)
        d = matrices[0].shape[0]

        companion = np.zeros((d * p, d * p), dtype=float)
        companion[:d, :] = np.hstack(matrices)
        if p > 1:
            # [I 0 ... 0; 0 I ... 0; ...] in the lower block rows.
            companion[d:, :-d] = np.eye(d * (p - 1))
        return companion

    @staticmethod
    def is_stable(mat: ArrayLike, *, tol: float = 0.0) -> bool:
        """Check the strict VAR stability condition ``rho(F) < 1``."""
        matrix = np.asarray(mat, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("stability matrix must be square")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("stability matrix must contain only finite values")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        radius = np.max(np.abs(np.linalg.eigvals(matrix)))
        return bool(radius < 1.0 - tol)

    def stationary_mean(self) -> np.ndarray:
        """Return the stationary mean of ``X_t`` for a stable VAR."""
        self._require_stable()
        lhs = np.eye(self.dim) - np.sum(self.coeffs, axis=0)
        return np.linalg.solve(lhs, self.bias)

    def stationary_state_cov(self) -> np.ndarray:
        """Return covariance of ``[X_t, X_{t-1}, ..., X_{t-p+1}]``.

        The covariance solves the discrete Lyapunov equation
        ``P = F P F.T + Q`` using a Kronecker linear system, avoiding an
        additional SciPy dependency.
        """
        self._require_stable()
        f = self.companion_matrix
        dp = self.dim * self.order
        q = np.zeros((dp, dp), dtype=float)
        q[: self.dim, : self.dim] = self.noise_cov

        # With NumPy's row-major vectorization:
        # vec_r(F P F^T) = (F \otimes F) vec_r(P).
        system = np.eye(dp * dp) - np.kron(f, f)
        p_vec = np.linalg.solve(system, q.reshape(-1))
        p_cov = p_vec.reshape(dp, dp)
        # Remove tiny numerical asymmetry.
        return 0.5 * (p_cov + p_cov.T)

    def sample_stationary_initial_history(self) -> np.ndarray:
        """Sample ``[X_{-p+1}, ..., X_0]`` from the stationary Gaussian law."""
        mean_x = self.stationary_mean()
        mean_state = np.tile(mean_x, self.order)
        cov_state = self.stationary_state_cov()
        state = self.rng.multivariate_normal(mean_state, cov_state)
        # state is [X_0, X_-1, ..., X_-p+1]; return chronological order.
        blocks = state.reshape(self.order, self.dim)
        return blocks[::-1].copy()

    def _require_stable(self) -> None:
        if not self.stable:
            raise ValueError(
                "stationary quantities are defined here only for a stable VAR "
                "(all companion eigenvalues must have modulus < 1)"
            )

    @staticmethod
    def _validate_coeffs(coeffs: Sequence[ArrayLike]) -> list[np.ndarray]:
        if coeffs is None or len(coeffs) == 0:
            raise ValueError("coeffs must contain at least one coefficient matrix")

        matrices = [np.asarray(coeff, dtype=float) for coeff in coeffs]
        first = matrices[0]
        if first.ndim != 2 or first.shape[0] != first.shape[1]:
            raise ValueError("every coefficient matrix must be square")
        d = first.shape[0]
        if d == 0:
            raise ValueError("coefficient matrices must be non-empty")

        for i, matrix in enumerate(matrices, start=1):
            if matrix.ndim != 2 or matrix.shape != (d, d):
                raise ValueError(
                    f"A{i} must have shape {(d, d)}; got {matrix.shape}"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"A{i} must contain only finite values")
        return [matrix.copy() for matrix in matrices]

    def _validate_vector(
        self, value: Optional[ArrayLike], name: str, *, default: np.ndarray
    ) -> np.ndarray:
        if value is None:
            return default.astype(float, copy=True)
        vector = np.asarray(value, dtype=float)
        if vector.shape != (self.dim,):
            raise ValueError(f"{name} must have shape {(self.dim,)}; got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector.copy()

    def _validate_init(self, init: ArrayLike) -> np.ndarray:
        history = np.asarray(init, dtype=float)
        expected = (self.order, self.dim)
        if history.shape != expected:
            raise ValueError(
                f"init must have shape {expected} and be ordered chronologically "
                f"as [X_-p+1, ..., X_0]; got {history.shape}"
            )
        if not np.all(np.isfinite(history)):
            raise ValueError("init must contain only finite values")
        return history.copy()

    def _validate_noise_cov(self, noise_cov: Optional[ArrayLike]) -> np.ndarray:
        if noise_cov is None:
            return np.eye(self.dim, dtype=float)
        cov = np.asarray(noise_cov, dtype=float)
        expected = (self.dim, self.dim)
        if cov.shape != expected:
            raise ValueError(f"noise_cov must have shape {expected}; got {cov.shape}")
        if not np.all(np.isfinite(cov)):
            raise ValueError("noise_cov must contain only finite values")
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
            raise ValueError("noise_cov must be symmetric")
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) < -1e-10:
            raise ValueError("noise_cov must be positive semidefinite")
        # Symmetrize tiny floating-point discrepancies.
        return 0.5 * (cov + cov.T)
