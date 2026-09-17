# Changelog

## 0.2.0

- Fixed innovation generation: one Gaussian shock is now drawn per time step,
  rather than one shock per lag.
- Fixed companion-matrix construction for arbitrary VAR order `p >= 1`.
- Added explicit innovation covariance `NOISE_COV` / `noise_cov`.
- Added `BIAS` support to YAML/CLI generation.
- Made `trajectory_length` mean exactly the number of returned/written points.
- Documented and validated chronological initial-state ordering.
- Added exact stationary Gaussian initialization for stable VARs.
- Added model/input validation for matrices, initial states, bias, covariance,
  and trajectory length.
- Changed the preferred YAML coefficient representation to an ordered list;
  legacy `A1`, `A2`, ... mappings are parsed by numeric lag rather than YAML
  insertion order.
- Added reproducible random-number generation via a seed or NumPy Generator.
- Added package metadata, CLI entry point, and declared Parquet dependency.
- Rewrote the README to state the mathematical model, stability criterion, and
  stationarity distinction explicitly.
- Added regression tests for the corrected mathematical and API behavior.
