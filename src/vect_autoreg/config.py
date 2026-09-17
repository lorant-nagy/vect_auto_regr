"""Configuration model for dataset generation."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

Matrix = List[List[float]]
CoeffConfig = Union[List[Matrix], Dict[str, Matrix]]


class Config(BaseModel):
    dest_folder: str
    sample_count: int = Field(gt=0)
    trajectory_length: int = Field(gt=0)
    COEFFS: CoeffConfig
    INIT: Optional[List[List[float]]] = None
    BIAS: Optional[List[float]] = None
    NOISE_COV: Optional[Matrix] = None
    stationary_init: bool = False
    seed: Optional[int] = None

    def ordered_coeffs(self) -> List[Matrix]:
        """Return coefficient matrices in explicit lag order.

        A YAML list is preferred. Legacy mappings ``A1:``, ``A2:``, ... are
        supported but sorted by their numeric suffix, so YAML insertion order
        cannot silently change the mathematical model.
        """
        if isinstance(self.COEFFS, list):
            if not self.COEFFS:
                raise ValueError("COEFFS must contain at least one matrix")
            return self.COEFFS

        parsed = []
        for key, matrix in self.COEFFS.items():
            match = re.fullmatch(r"A([1-9]\d*)", key)
            if match is None:
                raise ValueError(
                    "legacy COEFFS mappings must use keys A1, A2, ..., Ap; "
                    f"got {key!r}"
                )
            parsed.append((int(match.group(1)), matrix))

        if not parsed:
            raise ValueError("COEFFS must contain at least one matrix")
        parsed.sort(key=lambda item: item[0])
        expected = list(range(1, len(parsed) + 1))
        actual = [lag for lag, _ in parsed]
        if actual != expected:
            raise ValueError(
                f"legacy COEFFS keys must be consecutive A1..Ap; got lags {actual}"
            )
        return [matrix for _, matrix in parsed]
