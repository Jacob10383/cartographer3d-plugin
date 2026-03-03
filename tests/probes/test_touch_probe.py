from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cartographer.interfaces.configuration import Configuration, TouchModelConfiguration
from cartographer.interfaces.printer import Mcu, Position, TemperatureStatus, Toolhead
from cartographer.probe.touch_mode import TouchMode, TouchModeConfiguration

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from cartographer.probe.probe import Probe


@pytest.fixture(autouse=True)
def configure_probe(probe: Probe, config: Configuration) -> None:
    config.save_touch_model(
        TouchModelConfiguration(
            name="test_touch",
            speed=3,
            threshold=1000,
            z_offset=0,
        )
    )
    probe.touch.load_model("test_touch")


def test_probe_success(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0.5


def test_probe_includes_z_offset(
    mocker: MockerFixture, toolhead: Toolhead, config: Configuration, probe: Probe
) -> None:
    config.save_touch_model(
        TouchModelConfiguration(
            name="test_touch",
            speed=3,
            threshold=1000,
            z_offset=-0.5,
        )
    )
    probe.touch.load_model("test_touch")
    toolhead.z_probing_move = mocker.Mock(return_value=-0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0


def test_scan_pre_approach_runs_fast_move_when_not_triggered(
    mocker: MockerFixture,
    mcu: Mcu,
    toolhead: Toolhead,
    config: Configuration,
) -> None:
    """When scan says not triggered, perform_probe should do a fast z_probing_move via scan before touch."""
    from cartographer.probe.scan_mode import ScanMode

    scan = mocker.Mock(spec=ScanMode, autospec=True)
    scan.is_ready = True
    scan.query_is_triggered = mocker.Mock(return_value=False)
    scan.probe_speed = 5.0

    touch = TouchMode(
        mcu,
        toolhead,
        TouchModeConfiguration.from_config(config),
        scan_mode=scan,
    )
    touch.load_model("test_touch")

    # fast scan approach triggers at 2.0, then touch probes return 0.5
    toolhead.z_probing_move = mocker.Mock(side_effect=[2.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 5))

    _ = touch.perform_probe()

    # First call should be the scan pre-approach, rest are touch probes
    assert toolhead.z_probing_move.mock_calls[0] == mocker.call(scan, speed=5.0)
    assert toolhead.z_probing_move.mock_calls[1] == mocker.call(touch, speed=3)


def test_scan_pre_approach_skipped_when_already_triggered(
    mocker: MockerFixture,
    mcu: Mcu,
    toolhead: Toolhead,
    config: Configuration,
) -> None:
    """When scan says already triggered, no fast z_probing_move via scan should happen."""
    from cartographer.probe.scan_mode import ScanMode

    scan = mocker.Mock(spec=ScanMode, autospec=True)
    scan.is_ready = True
    scan.query_is_triggered = mocker.Mock(return_value=True)

    touch = TouchMode(
        mcu,
        toolhead,
        TouchModeConfiguration.from_config(config),
        scan_mode=scan,
    )
    touch.load_model("test_touch")

    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 3))

    _ = touch.perform_probe()

    # All probing moves should be touch probes, none via scan
    for call in toolhead.z_probing_move.mock_calls:
        assert call == mocker.call(touch, speed=3)


def test_probe_moves_below_2(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))
    move_spy = mocker.spy(toolhead, "move")

    _ = probe.touch.perform_probe()

    assert move_spy.mock_calls[0] == mocker.call(z=2, speed=mocker.ANY)


def test_does_not_move_above_2(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(return_value=0.5)
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 10))
    move_spy = mocker.spy(toolhead, "move")

    _ = probe.touch.perform_probe()

    assert move_spy.mock_calls[0] != mocker.call(z=2, speed=mocker.ANY)


def test_probe_standard_deviation_failure(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(side_effect=[1 + i * 0.1 for i in range(20)])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()


def test_probe_suceeds_on_more(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.z_probing_move = mocker.Mock(side_effect=[1.0, 1.01, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0.5


def test_probe_spread_samples_rejected_by_window(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    # Spread-out good samples interleaved with bad ones can no longer be cherry-picked
    # because the sliding window only considers the most recent samples + max_noisy_samples.
    toolhead.z_probing_move = mocker.Mock(side_effect=[0.5, 1.0, 1.5, 0.5, 2.5, 0.5, 3.5, 0.5, 4.5, 0.5, 5.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    with pytest.raises(RuntimeError, match="Unable to find"):
        _ = probe.touch.perform_probe()


def test_probe_succeeds_within_window(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    # First 3 are noisy, then 5 consistent samples within the window.
    # samples=5, max_noisy_samples=2: window=7, so the last 5 samples [0.5, 0.5, 0.5, 0.5, 0.5] all agree.
    toolhead.z_probing_move = mocker.Mock(side_effect=[1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    toolhead.get_position = mocker.Mock(return_value=Position(0, 0, 1))

    assert probe.touch.perform_probe() == 0.5


def test_probe_unhomed_z(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.is_homed = mocker.Mock(return_value=False)

    with pytest.raises(RuntimeError, match="Z axis must be homed"):
        _ = probe.touch.perform_probe()


def test_home_wait(mocker: MockerFixture, mcu: Mcu, probe: Probe) -> None:
    mcu.stop_homing = mocker.Mock(return_value=1.5)

    assert probe.touch.home_wait(home_end_time=1.0) == 1.5


def test_abort_if_current_extruder_too_hot(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_extruder_temperature = mocker.Mock(return_value=TemperatureStatus(156, 0))

    with pytest.raises(RuntimeError, match="Nozzle temperature must be below 150C"):
        _ = probe.touch.home_start(print_time=0.0)


def test_abort_if_current_extruder_target_too_hot(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_extruder_temperature = mocker.Mock(return_value=TemperatureStatus(0, 156))

    with pytest.raises(RuntimeError, match="Nozzle temperature must be below 150C"):
        _ = probe.touch.home_start(print_time=0.0)


def test_nozzle_outside_bounds(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_position = mocker.Mock(return_value=Position(-10, 0, 1))

    with pytest.raises(RuntimeError, match="outside .* boundaries"):
        _ = probe.touch.home_start(0)


def test_probe_outside_bounds(mocker: MockerFixture, toolhead: Toolhead, probe: Probe) -> None:
    toolhead.get_position = mocker.Mock(return_value=Position(295, 95, 1))

    with pytest.raises(RuntimeError, match="outside .* boundaries"):
        _ = probe.touch.home_start(0)
