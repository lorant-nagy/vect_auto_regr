from pathlib import Path

import pandas as pd

from vect_autoreg.config import Config
from vect_autoreg.generate import generate


def test_generate_writes_exact_trajectory_length_and_passes_bias_covariance(tmp_path, monkeypatch):
    written = []

    def fake_to_parquet(self, path, index=False):
        written.append((Path(path), self.copy(), index))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    config = Config(
        dest_folder=str(tmp_path),
        sample_count=2,
        trajectory_length=7,
        seed=7,
        stationary_init=False,
        COEFFS=[[[0.2]], [[0.1]]],
        BIAS=[0.5],
        NOISE_COV=[[1.2]],
        INIT=[[0.0], [0.0]],
    )

    dest, _, stable = generate(config)
    assert stable
    assert Path(dest).exists()
    assert len(written) == 2
    for path, frame, index in written:
        assert path.suffix == ".parquet"
        assert len(frame) == 7
        assert list(frame.columns) == ["time", "x_1"]
        assert index is False
