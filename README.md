# Vector autoregression simulator

A small NumPy implementation of a Gaussian vector autoregressive process.

For a VAR(`p`) in `R^d`, the model is

\[
X_t = c + \sum_{j=1}^p A_j X_{t-j} + \varepsilon_t,
\qquad
\varepsilon_t \sim N(0, \Sigma),
\]

with independent innovations at different time steps.

The implementation is in `autoreg.py` and only depends on NumPy.

## Example

```python
import numpy as np
from autoreg import VectAutoReg

A1 = np.array([[0.5, 0.1],
               [0.1, 0.5]])
A2 = np.array([[0.2, 0.0],
               [0.0, 0.2]])

model = VectAutoReg([A1, A2], seed=1)

print(model.is_stable())
series = model.simulate(1000)
```

`series` has shape `(1000, 2)` in this example.

## Stability

The code forms the usual companion matrix

\[
F =
\begin{pmatrix}
A_1 & A_2 & \cdots & A_p \\
I   & 0   & \cdots & 0 \\
0   & I   & \cdots & 0 \\
\vdots & & \ddots & \vdots \\
0 & \cdots & I & 0
\end{pmatrix}.
\]

The VAR is stable when every eigenvalue of `F` has absolute value smaller than 1.

```python
model.is_stable()
```

works for arbitrary order `p`.

## Initial values and stationarity

If no initial values are supplied, the process starts from zeros. For a VAR(`p`), custom initial values should be supplied chronologically as

\[
[X_{-p+1},\ldots,X_0].
\]

For example, for a two-dimensional VAR(2):

```python
init = np.array([
    [1.0, 2.0],   # X_-1
    [3.0, 4.0],   # X_0
])
```

A stable VAR started from fixed initial values is not exactly stationary at the beginning. If a nearly stationary sample is wanted, discard an initial burn-in period:

```python
series = model.simulate(1000, burn_in=500)
```

## Innovation covariance and bias

By default,

\[
c=0, \qquad \Sigma=I.
\]

Both can be changed:

```python
bias = np.array([0.1, -0.2])
Sigma = np.array([[1.0, 0.3],
                  [0.3, 2.0]])

model = VectAutoReg(
    [A1, A2],
    bias=bias,
    noise_cov=Sigma,
    seed=1,
)
```

There is exactly one Gaussian innovation vector per time step, independent of the autoregressive order.
