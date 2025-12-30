from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cartographer.coil.temperature_compensation import CoilTemperatureCompensationModel
from cartographer.interfaces.configuration import CoilCalibrationConfiguration


class _MockMcu:
    def get_coil_reference(self):
        return SimpleNamespace(min_frequency=1000.0, min_frequency_temperature=25.0)


def test_compensate_batch_matches_scalar() -> None:
    model = CoilTemperatureCompensationModel(
        CoilCalibrationConfiguration(
            a_a=1e-8,
            a_b=1e-5,
            b_a=2e-8,
            b_b=1e-4,
        ),
        _MockMcu(),
    )
    rng = np.random.default_rng(0)
    frequencies = rng.uniform(900.0, 1100.0, size=256)
    temp_sources = rng.uniform(20.0, 300.0, size=256)
    temp_target = 140.0

    scalar = np.array(
        [
            model.compensate(float(frequency), float(temp_source), temp_target)
            for frequency, temp_source in zip(frequencies, temp_sources)
        ]
    )
    batch = model.compensate_batch(frequencies, temp_sources, temp_target)

    np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-12)
