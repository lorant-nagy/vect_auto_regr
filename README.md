# Vector Autoregression

Simple simulation of a VAR(\(p\)) process

$$
X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t,
\qquad
\varepsilon_t \sim N(0,\Sigma).
$$

Stability is checked using the companion matrix. The process is stable if all eigenvalues have modulus smaller than \(1\).

## Example

```python
import numpy as np
from autoreg import VectAutoReg

A1 = np.array([
    [0.5, 0.1],
    [0.1, 0.5]
])

A2 = np.array([
    [0.2, 0.0],
    [0.0, 0.2]
])

model = VectAutoReg([A1, A2], seed=1)

print("Stable:", model.is_stable())

series = model.simulate(1000)
print(series.shape)
```

For a stable VAR, a burn-in can be used to reduce the effect of the initial condition:

```python
series = model.simulate(1000, burn_in=500)
```
