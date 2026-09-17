import numpy as np


class VectAutoReg:
    """Simple Gaussian VAR(p) simulator.

    X_t = c + A_1 X_{t-1} + ... + A_p X_{t-p} + eps_t,
    where eps_t ~ N(0, Sigma).
    """

    def __init__(self, coeffs, bias=None, init=None, noise_cov=None, seed=None):
        self.coeffs = [np.asarray(A, dtype=float) for A in coeffs]
        if len(self.coeffs) == 0:
            raise ValueError("At least one coefficient matrix is required.")

        self.order = len(self.coeffs)
        self.dim = self.coeffs[0].shape[0]

        for A in self.coeffs:
            if A.shape != (self.dim, self.dim):
                raise ValueError("All coefficient matrices must have the same square shape.")

        self.bias = (
            np.zeros(self.dim)
            if bias is None
            else np.asarray(bias, dtype=float)
        )
        if self.bias.shape != (self.dim,):
            raise ValueError("bias must have shape (dimension,).")

        self.noise_cov = (
            np.eye(self.dim)
            if noise_cov is None
            else np.asarray(noise_cov, dtype=float)
        )
        if self.noise_cov.shape != (self.dim, self.dim):
            raise ValueError("noise_cov must have shape (dimension, dimension).")
        if not np.allclose(self.noise_cov, self.noise_cov.T):
            raise ValueError("noise_cov must be symmetric.")
        if np.min(np.linalg.eigvalsh(self.noise_cov)) < -1e-12:
            raise ValueError("noise_cov must be positive semidefinite.")

        self.init = (
            np.zeros((self.order, self.dim))
            if init is None
            else np.asarray(init, dtype=float)
        )
        if self.init.shape != (self.order, self.dim):
            raise ValueError(
                "init must have shape (order, dimension), ordered as "
                "[X_{-p+1}, ..., X_0]."
            )

        self.rng = np.random.default_rng(seed)

    def companion_matrix(self):
        """Return the standard companion matrix of the VAR."""
        p, d = self.order, self.dim
        F = np.zeros((p * d, p * d))
        F[:d, :] = np.hstack(self.coeffs)
        if p > 1:
            F[d:, :-d] = np.eye((p - 1) * d)
        return F

    def is_stable(self):
        """Check whether every companion-matrix eigenvalue has modulus < 1."""
        eigenvalues = np.linalg.eigvals(self.companion_matrix())
        return np.all(np.abs(eigenvalues) < 1)

    def simulate(self, length, burn_in=0):
        """Simulate ``length`` observations after an optional burn-in.

        The supplied initial history is used to start the recursion.  If
        ``burn_in`` is positive, those generated observations are discarded.
        """
        if length <= 0 or burn_in < 0:
            raise ValueError("length must be positive and burn_in non-negative.")

        history = [x.copy() for x in self.init]
        total_steps = burn_in + length
        output = []

        for step in range(total_steps):
            x = self.bias.copy()

            for lag, A in enumerate(self.coeffs, start=1):
                x += A @ history[-lag]

            # Exactly one innovation vector is added at each time step.
            x += self.rng.multivariate_normal(np.zeros(self.dim), self.noise_cov)

            history.append(x)
            if len(history) > self.order:
                history.pop(0)

            if step >= burn_in:
                output.append(x.copy())

        return np.asarray(output)
