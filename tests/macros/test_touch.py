from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from typing_extensions import TypeAlias

from cartographer.interfaces.printer import MacroParams, Position, Toolhead
from cartographer.macros.touch import TouchAccuracyMacro, TouchHomeMacro, TouchProbeMacro
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration
from tests.mocks.params import MockParams

if TYPE_CHECKING:
    from pytest import LogCaptureFixture
    from pytest_mock import MockerFixture


Probe: TypeAlias = TouchMode


@pytest.fixture
def offset() -> Position:
    return Position(0, 0, 0)


@pytest.fixture
def probe(mocker: MockerFixture, offset: Position) -> Probe:
    mock = mocker.Mock(spec=Probe, autospec=True)
    mock.config = mocker.Mock(spec=TouchModeConfiguration, autospec=True)
    mock.config.move_speed = 42
    mock.offset = offset
    return mock


def test_touch_macro_output(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    params: MacroParams,
    toolhead: Toolhead,
):
    macro = TouchProbeMacro(probe, toolhead)
    probe.perform_probe = mocker.Mock(return_value=5.0)

    with caplog.at_level(logging.INFO):
        macro.run(params)

    assert "Result: at 10.000,10.000 estimate contact at z=5.000000" in caplog.messages


def test_touch_accuracy_macro_output(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    macro = TouchAccuracyMacro(probe, toolhead, lift_speed=5)
    params.get_int = mocker.Mock(return_value=10)
    toolhead.get_position = lambda: Position(0, 0, 0)
    params.get_float = mocker.Mock(return_value=1)
    i = -1
    measurements: list[float] = [50 + i * 10 for i in range(10)]

    def mock_probe(**_) -> float:
        nonlocal i
        i += 1
        return measurements[i]

    probe.perform_probe = mock_probe

    with caplog.at_level(logging.INFO):
        macro.run(params)

    assert "touch accuracy results" in caplog.text
    assert "minimum 50" in caplog.text
    assert "maximum 140" in caplog.text
    assert "range 90" in caplog.text
    assert "average 95" in caplog.text
    assert "median 95" in caplog.text
    assert "standard deviation 28" in caplog.text


def test_touch_accuracy_macro_sample_count(
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    macro = TouchAccuracyMacro(probe, toolhead, lift_speed=5)
    params.get_int = mocker.Mock(return_value=3)
    toolhead.get_position = lambda: Position(0, 0, 0)
    params.get_float = mocker.Mock(return_value=1)
    i = -1
    measurements: list[float] = [50 + i * 10 for i in range(10)]

    def mock_probe(**_) -> float:
        nonlocal i
        i += 1
        return measurements[i]

    probe.perform_probe = mock_probe

    with caplog.at_level(logging.INFO):
        macro.run(params)

    assert "touch accuracy results" in caplog.text
    assert "minimum 50" in caplog.text
    assert "maximum 70" in caplog.text
    assert "range 20" in caplog.text
    assert "average 60" in caplog.text
    assert "median 60" in caplog.text
    assert "standard deviation 8" in caplog.text


def test_touch_home_macro_moves(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=0.1)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    move_spy = mocker.spy(toolhead, "move")

    macro.run(params)

    assert move_spy.mock_calls == [
        mocker.call(z=4, speed=mocker.ANY),
        mocker.call(x=10, y=10, speed=mocker.ANY),
    ]


def test_touch_home_macro(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    # We are at 2, and touch the bed at -0.1.
    height = 2
    trigger = 0.1
    # That means that the bed was further away than we thought,
    # so we need to move the z axis "down".
    expected = height - trigger

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, height))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(expected)]


def test_touch_home_macro_applies_thermal_delta_with_coeff(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
) -> None:
    params = MockParams()
    params.params["PRINT_TEMP"] = "260"

    trigger = 0.1
    coeff = 0.0004
    measured_touch_temp = 140.0
    expected_delta = coeff * (260 - measured_touch_temp)
    expected = 2 - (trigger + expected_delta)

    model = mocker.Mock()
    model.name = "default"
    model.z_offset = 0.0
    model.thermal_expansion_coefficient = coeff
    probe.get_model = mocker.Mock(return_value=model)

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    toolhead.get_extruder_temperature = mocker.Mock(return_value=mocker.Mock(current=measured_touch_temp, target=140.0))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(expected)]


def test_touch_home_macro_thermal_and_z_offset_interaction(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
) -> None:
    params = MockParams()
    params.params["PRINT_TEMP"] = "260"

    # perform_probe() already returns trigger corrected by -z_offset.
    trigger = 0.625
    coeff = 0.0004
    z_offset = -0.2
    measured_touch_temp = 140.0
    expected_delta = coeff * (260 - measured_touch_temp)
    expected = 2 - (trigger + expected_delta)

    model = mocker.Mock()
    model.name = "default"
    model.z_offset = z_offset
    model.thermal_expansion_coefficient = coeff
    probe.get_model = mocker.Mock(return_value=model)

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    toolhead.get_extruder_temperature = mocker.Mock(return_value=mocker.Mock(current=measured_touch_temp, target=140.0))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(expected)]


def test_touch_home_macro_applies_negative_thermal_term_when_print_temp_lower(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
) -> None:
    params = MockParams()
    params.params["PRINT_TEMP"] = "200"

    trigger = 0.2
    coeff = 0.0004
    measured_touch_temp = 240.0
    expected_thermal_term = coeff * (200 - measured_touch_temp)
    expected = 2 - (trigger + expected_thermal_term)

    model = mocker.Mock()
    model.name = "default"
    model.z_offset = 0.0
    model.thermal_expansion_coefficient = coeff
    probe.get_model = mocker.Mock(return_value=model)

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    toolhead.get_extruder_temperature = mocker.Mock(return_value=mocker.Mock(current=measured_touch_temp, target=200.0))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(expected)]


def test_touch_home_macro_skips_thermal_if_coeff_missing(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
) -> None:
    params = MockParams()
    params.params["PRINT_TEMP"] = "260"

    trigger = 0.1
    expected = 2 - trigger

    model = mocker.Mock()
    model.name = "default"
    model.z_offset = 0.0
    model.thermal_expansion_coefficient = None
    probe.get_model = mocker.Mock(return_value=model)

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    toolhead.get_extruder_temperature = mocker.Mock(return_value=mocker.Mock(current=140.0, target=140.0))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(expected)]


def test_unhomed_touch_home_macro(
    mocker: MockerFixture,
    probe: Probe,
    toolhead: Toolhead,
    params: MacroParams,
):
    toolhead.is_homed = lambda axis: axis != "z"
    max_offset = 10
    height = toolhead.get_axis_limits("z")[1] - max_offset
    trigger = 0.1
    # That means that the bed was further away than we thought,
    # so we need to move the z axis "down".
    expected = height - trigger

    macro = TouchHomeMacro(probe, toolhead, home_position=(10, 10), lift_speed=5, travel_speed=50, random_radius=0)
    probe.perform_probe = mocker.Mock(return_value=trigger)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, height))
    set_z_position_spy = mocker.spy(toolhead, "set_z_position")

    macro.run(params)

    assert set_z_position_spy.mock_calls == [mocker.call(z=height), mocker.call(expected)]


@pytest.mark.parametrize(
    "u1, u2, expected_x, expected_y",
    [
        (0.0, 0.0, 50.0, 50.0),  # Center point
        (1.0, 0.0, 60.0, 50.0),  # Max radius, 0 angle
        (0.25, 0.25, 50.0, 55.0),  # radius=5, angle=π/2
        (0.16, 0.5, 46.0, 50.0),  # radius=4, angle=π
    ],
)
def test_random_radius_uniform_distribution(
    mocker: MockerFixture,
    probe: TouchMode,
    toolhead: Toolhead,
    params: MacroParams,
    u1: float,
    u2: float,
    expected_x: float,
    expected_y: float,
):
    """Test that random positions are generated correctly with square root method."""
    macro = TouchHomeMacro(probe, toolhead, home_position=(50, 50), lift_speed=5, travel_speed=50, random_radius=10.0)
    probe.perform_probe = mocker.Mock(return_value=0.1)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 2))
    move_spy = mocker.spy(toolhead, "move")

    _ = mocker.patch("cartographer.macros.touch.home.random", side_effect=[u1, u2])
    macro.run(params)

    assert move_spy.mock_calls == [
        mocker.call(z=4, speed=mocker.ANY),
        mocker.call(x=expected_x, y=expected_y, speed=mocker.ANY),
    ]
