import pytest

from vect_autoreg.config import Config


def base_config(coeffs):
    return Config(
        dest_folder="./data",
        sample_count=1,
        trajectory_length=5,
        COEFFS=coeffs,
    )


def test_legacy_dict_coefficients_are_sorted_by_lag_number():
    a1 = [[0.1]]
    a2 = [[0.2]]
    config = base_config({"A2": a2, "A1": a1})
    assert config.ordered_coeffs() == [a1, a2]


def test_legacy_dict_coefficients_must_be_consecutive():
    config = base_config({"A1": [[0.1]], "A3": [[0.3]]})
    with pytest.raises(ValueError, match="consecutive"):
        config.ordered_coeffs()


def test_list_coefficients_preserve_explicit_order():
    coeffs = [[[0.1]], [[0.2]]]
    config = base_config(coeffs)
    assert config.ordered_coeffs() == coeffs
