# vect_auto_regr

A small simulator for Gaussian vector autoregressive processes.

## Mathematical model

For `X_t` in `R^d`, a VAR(`p`) is

\[
X_t = c + \sum_{j=1}^{p} A_j X_{t-j} + \varepsilon_t,
\qquad
\varepsilon_t \stackrel{iid}{\sim} N(0,\Sigma),
\]

where every `A_j` is a `d x d` matrix, `c` is an optional intercept, and
`Sigma` is the innovation covariance. The implementation adds **one** innovation
vector per time step, independent of the autoregressive order.

Given an initial history

\[
X_{-p+1},\ldots,X_0,
\]

the recursion uniquely determines future observations.

### Stability

The standard companion matrix is

\[
F =
\begin{pmatrix}
A_1 & A_2 & \cdots & A_p\\
I   & 0   & \cdots & 0\\
0   & I   & \cdots & 0\\
\vdots & & \ddots & \vdots\\
0 & \cdots & I & 0
\end{pmatrix}.
\]

The VAR is stable exactly when every eigenvalue of `F` has modulus strictly
smaller than 1. For a stable Gaussian VAR, there is a unique causal stationary
Gaussian solution.

A stable model started from arbitrary deterministic values is **not** stationary
at the beginning; it only approaches the stationary regime. This package
therefore supports two explicit initialization modes:

- provide `init=[X_-p+1, ..., X_0]`; or
- omit `init` and call with `stationary_init=True` to sample the initial history
  from the exact stationary Gaussian law.

## Installation

```bash
pip install -e .
```

For development/tests:

```bash
pip install -e '.[test]'
pytest
```

The declared dependencies include `pyarrow`, which is needed for the CLI's
Parquet output.

## Python usage

```python
import numpy as np
from vect_autoreg import VectAutoReg

A1 = np.array([[0.5, 0.1], [0.1, 0.5]])
A2 = np.array([[0.2, 0.0], [0.0, 0.2]])

model = VectAutoReg(
    [A1, A2],
    bias=np.array([0.1, -0.2]),
    noise_cov=np.array([[1.0, 0.3], [0.3, 2.0]]),
    rng=1234,
)

# Exactly 1000 returned time points. Because this is VAR(2), the first two
# points are the initial history (zeros here), followed by 998 simulated points.
series = model(1000)
assert series.shape == (2, 1000)

print(model.companion_matrix)
print(model.stable)
```

To obtain a trajectory that is stationary from its first returned point:

```python
stationary_series = model(1000, stationary_init=True)
```

`stationary_init=True` requires a stable model and cannot be combined with a
user-supplied `init`.

### Initial-condition ordering

For a VAR(2),

```python
init = np.array([
    [1.0, 2.0],  # X_-1
    [3.0, 4.0],  # X_0
])
```

is chronological. The next observation is

\[
X_1 = c + A_1 X_0 + A_2 X_{-1} + \varepsilon_1.
\]

## YAML dataset generation

See `config/config.yaml`. Coefficients should preferably be written as an
**ordered list** `[A1, A2, ..., Ap]`, so lag order is part of the data structure:

```yaml
dest_folder: ./data
sample_count: 10
trajectory_length: 100
seed: 12345
stationary_init: false

COEFFS:
  - - [0.5, 0.1]
    - [0.1, 0.5]
  - - [0.2, 0.0]
    - [0.0, 0.2]

BIAS: [0.0, 0.0]
NOISE_COV:
  - [1.0, 0.2]
  - [0.2, 1.0]

INIT:
  - [0.0, 0.0]
  - [0.0, 0.0]
```

`trajectory_length` is the **total number of rows returned/written**, including
the `p` initial observations. It must therefore be at least `p`.

Legacy mappings such as `A1:`, `A2:` are still accepted, but their numeric lag
suffixes are parsed and sorted explicitly; YAML insertion order cannot silently
change the model.

Generate datasets with either

```bash
vect-autoreg-generate --config_path config/config.yaml
```

or

```bash
python generate.py --config_path config/config.yaml
```

Each trajectory is written to a Parquet file, and the resolved configuration
plus the stability result are written under the generated run's `log/` folder.

## Validation

Construction rejects malformed mathematical inputs, including:

- empty, non-square, mismatched, or non-finite coefficient matrices;
- incorrectly shaped initial histories or bias vectors;
- non-symmetric or non-positive-semidefinite innovation covariance matrices;
- stationary initialization for unstable models.

The test suite includes arbitrary-order companion matrices, the one-innovation-
per-step property, exact trajectory-length semantics, stationary initialization,
and legacy coefficient-order handling.
