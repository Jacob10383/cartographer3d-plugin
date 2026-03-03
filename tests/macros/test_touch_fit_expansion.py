from __future__ import annotations

import pytest

from cartographer.macros.touch.fit_expansion import (
    build_temperature_sweep,
    fit_linear_expansion,
)


def test_build_temperature_sweep_ascending() -> None:
    assert build_temperature_sweep(120, 150, 10) == [120, 130, 140, 150]


def test_build_temperature_sweep_descending() -> None:
    assert build_temperature_sweep(200, 170, 10) == [200, 190, 180, 170]


def test_build_temperature_sweep_appends_end_when_not_divisible() -> None:
    assert build_temperature_sweep(120, 145, 10) == [120, 130, 140, 145]


def test_build_temperature_sweep_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError, match="TEMP_STEP"):
        _ = build_temperature_sweep(120, 160, 0)


def test_fit_linear_expansion_returns_expected_line() -> None:
    slope, intercept, r_squared = fit_linear_expansion(
        [
            (100.0, 0.200),
            (120.0, 0.220),
            (140.0, 0.240),
        ]
    )

    assert slope == pytest.approx(0.001)  # pyright: ignore[reportUnknownMemberType]
    assert intercept == pytest.approx(0.100)  # pyright: ignore[reportUnknownMemberType]
    assert r_squared == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


def test_fit_linear_expansion_requires_two_points() -> None:
    with pytest.raises(ValueError, match="At least two"):
        _ = fit_linear_expansion([(120.0, 0.200)])
