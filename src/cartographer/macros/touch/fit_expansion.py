from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Sequence, final

import numpy as np
from typing_extensions import override

from cartographer.interfaces.printer import GCodeDispatch, Macro, MacroParams, Toolhead
from cartographer.macros.fields import param, parse

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.probe.touch_mode import TouchMode

logger = logging.getLogger(__name__)


DEFAULT_DUMP_DIR = "/mnt/UDISK/root/carto_dumps"


class SweepMode(str, Enum):
    UP = "up"
    UP_DOWN = "up_down"


def _parse_positive_speed_override(params: MacroParams) -> float | None:
    return params.get_float("SPEED", default=None, above=0)


def _parse_probe_point(params: MacroParams) -> tuple[float, float] | None:
    raw = params.get("PROBE_POINT", default=None)
    if raw is None:
        raw = params.get("probe_point", default=None)
    if raw is None:
        return None

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        msg = "PROBE_POINT must be formatted as X,Y (example: PROBE_POINT=175,175)"
        raise RuntimeError(msg)

    try:
        x = float(parts[0])
        y = float(parts[1])
    except ValueError as exc:
        msg = f"Invalid PROBE_POINT '{raw}'. Expected numeric X,Y values."
        raise RuntimeError(msg) from exc
    return (x, y)


def build_temperature_sweep(start_temp: int, end_temp: int, step: int) -> list[int]:
    """Build an inclusive temperature sweep from start to end."""
    if step <= 0:
        msg = "TEMP_STEP must be > 0"
        raise ValueError(msg)

    if start_temp == end_temp:
        return [start_temp]

    direction = 1 if end_temp > start_temp else -1
    signed_step = step * direction
    temps = list(range(start_temp, end_temp + direction, signed_step))
    if temps[-1] != end_temp:
        temps.append(end_temp)
    return temps


def build_temperature_sequence(temperatures: Sequence[int], mode: SweepMode) -> list[tuple[str, int]]:
    """Build a sweep sequence with direction labels."""
    if not temperatures:
        return []

    upward = [("up", t) for t in temperatures]
    if mode == SweepMode.UP or len(temperatures) == 1:
        return upward

    # Downward leg excludes the peak duplicate but includes the start temperature.
    downward = [("down", t) for t in temperatures[-2::-1]]
    return upward + downward


def fit_linear_expansion(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    """Fit y = slope*x + intercept and return (slope, intercept, r_squared)."""
    if len(points) < 2:
        msg = "At least two temperature points are required to fit expansion"
        raise ValueError(msg)

    temperatures = np.array([p[0] for p in points], dtype=float)
    medians = np.array([p[1] for p in points], dtype=float)

    slope, intercept = np.polyfit(temperatures, medians, 1)
    fitted = slope * temperatures + intercept
    residual = medians - fitted

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((medians - np.mean(medians)) ** 2))
    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

    return float(slope), float(intercept), r_squared


@dataclass(frozen=True)
class TouchFitExpansionParams:
    """Parameters for CARTOGRAPHER_TOUCH_FIT_EXPANSION."""

    start_temp: int = param("Nozzle temperature to start at (°C)", key="START_TEMP", min=20, max=350)
    end_temp: int = param("Nozzle temperature to end at (°C)", key="END_TEMP", min=20, max=350)
    temp_step: int = param("Temperature step size (°C)", key="TEMP_STEP", default=10, min=1, max=100)
    sweep_mode: SweepMode = param("Sweep mode: up or up_down", default=SweepMode.UP, key="SWEEP_MODE")
    samples: int = param("Probe runs per temperature", default=5, min=1, max=50)
    dwell: float = param("Extra settle time per temperature in seconds", default=15.0, min=0.0, max=120.0)
    tolerance: float = param("TEMPERATURE_WAIT tolerance (+/- °C)", default=1.0, min=0.1, max=20.0)
    stability_seconds: float = param(
        "Require near-steady nozzle temperature for this long before probing.",
        default=6.0,
        min=0.0,
        max=120.0,
        key="STABILITY_SECONDS",
    )
    stability_rate_limit: float = param(
        "Maximum allowed temp drift in °C/s during stability window.",
        default=0.18,
        min=0.01,
        max=5.0,
        key="STABILITY_RATE_LIMIT",
    )
    stability_timeout: float = param(
        "Max seconds to wait for stability after TEMPERATURE_WAIT.",
        default=180.0,
        min=1.0,
        max=3600.0,
        key="STABILITY_TIMEOUT",
    )
    clean_above: int = param(
        "Run NOZZLE_CLEAN before each sample when setpoint is above this temperature (°C)",
        default=200,
        min=0,
        max=350,
        key="CLEAN_ABOVE",
    )
    post_clean_dwell: float = param(
        "Extra settle time in seconds after clean and stability checks.",
        default=2.0,
        min=0.0,
        max=120.0,
        key="POST_CLEAN_DWELL",
    )
    probe_point: tuple[float, float] | None = param(
        "Optional fixed probing XY point as X,Y (example: 175,175)",
        default=None,
        key="PROBE_POINT",
        parse_fn=_parse_probe_point,
    )
    model: str | None = param("Touch model to fit and update", default=None)
    apply: bool = param("Save fit to the selected touch model", default=True)
    threshold: int | None = param("Threshold override for probing", default=None, key="THRESHOLD", min=1)
    speed: float | None = param(
        "Probe speed override for fitting in mm/s",
        default=None,
        key="SPEED",
        parse_fn=_parse_positive_speed_override,
    )
    unsafe_override_temp_limit: bool = param(
        "Ignore configured touch max temperature limit during fitting",
        default=False,
        key="UNSAFE_OVERRIDE_TEMP_LIMIT",
    )


@dataclass(frozen=True)
class TemperatureFitResult:
    sequence_index: int
    direction: str
    setpoint: int
    average_temp: float
    probe_median: float
    probe_stddev: float
    probe_min: float
    probe_max: float
    count: int


@final
class TouchFitExpansionMacro(Macro):
    description = "Fit touch thermal expansion vs nozzle temperature and dump raw data."

    def __init__(
        self,
        probe: TouchMode,
        toolhead: Toolhead,
        config: Configuration,
        gcode: GCodeDispatch,
    ) -> None:
        self._probe = probe
        self._toolhead = toolhead
        self._config = config
        self._gcode = gcode

    @override
    def run(self, params: MacroParams) -> None:
        p = parse(TouchFitExpansionParams, params)
        self._validate_homing()
        self._ensure_model_loaded(p.model)

        temperatures = build_temperature_sweep(p.start_temp, p.end_temp, p.temp_step)
        if len(temperatures) < 2:
            msg = "Temperature sweep must contain at least two unique temperatures"
            raise RuntimeError(msg)

        self._validate_temperature_limits(temperatures, p.unsafe_override_temp_limit)
        sequence = build_temperature_sequence(temperatures, p.sweep_mode)

        model_name = self._probe.get_model().name
        raw_dump_path, summary_dump_path = self._build_dump_paths(model_name)
        effective_probe_point = p.probe_point
        if effective_probe_point is None:
            position = self._toolhead.get_position()
            effective_probe_point = (position.x, position.y)

        logger.info(
            "Starting touch expansion fit for model '%s' over %s (mode=%s)",
            model_name,
            ", ".join(str(t) for t in temperatures),
            p.sweep_mode.value,
        )
        if p.probe_point is not None:
            logger.info("Using fixed probe point at X=%.3f Y=%.3f", effective_probe_point[0], effective_probe_point[1])
        else:
            logger.info(
                "Using current toolhead XY as probe point at X=%.3f Y=%.3f",
                effective_probe_point[0],
                effective_probe_point[1],
            )

        logger.info("Raw dump: %s", raw_dump_path)
        logger.info("Summary dump: %s", summary_dump_path)
        self._init_raw_dump(raw_dump_path)
        self._init_summary_progress_dump(summary_dump_path)

        fit_results: list[TemperatureFitResult] = []

        for sequence_index, (direction, setpoint) in enumerate(sequence, start=1):
            self._go_to_wastebin()
            self._set_nozzle_temperature(setpoint, p.tolerance)
            self._wait_for_stability(
                setpoint=setpoint,
                tolerance=p.tolerance,
                stability_seconds=p.stability_seconds,
                rate_limit=p.stability_rate_limit,
                timeout=p.stability_timeout,
            )
            if p.dwell > 0:
                self._toolhead.dwell(p.dwell)
            if setpoint <= p.clean_above:
                self._move_to_probe_point(effective_probe_point)

            probes: list[float] = []
            nozzle_temps: list[float] = []
            for sample_index in range(1, p.samples + 1):
                if setpoint > p.clean_above:
                    self._go_to_wastebin()
                    self._run_nozzle_clean()
                    self._wait_for_stability(
                        setpoint=setpoint,
                        tolerance=p.tolerance,
                        stability_seconds=p.stability_seconds,
                        rate_limit=p.stability_rate_limit,
                        timeout=p.stability_timeout,
                    )
                    if p.post_clean_dwell > 0:
                        self._toolhead.dwell(p.post_clean_dwell)
                    self._move_to_probe_point(effective_probe_point)

                probe_value = self._probe.perform_probe(
                    p.threshold,
                    p.speed,
                    ignore_temp_limit=p.unsafe_override_temp_limit,
                )
                status = self._toolhead.get_extruder_temperature()
                probes.append(probe_value)
                nozzle_temps.append(status.current)
                self._append_raw_dump_row(
                    raw_dump_path,
                    (
                        setpoint,
                        sample_index,
                        status.current,
                        status.target,
                        probe_value,
                        direction,
                        sequence_index,
                    ),
                )

            summary = TemperatureFitResult(
                sequence_index=sequence_index,
                direction=direction,
                setpoint=setpoint,
                average_temp=float(np.mean(nozzle_temps)),
                probe_median=float(np.median(probes)),
                probe_stddev=float(np.std(probes)),
                probe_min=min(probes),
                probe_max=max(probes),
                count=len(probes),
            )
            fit_results.append(summary)
            self._append_summary_progress_row(summary_dump_path, summary)
            logger.info(
                "[%s] Temp %d°C: median=%.6f std=%.6f range=%.6f (%d samples)",
                direction,
                setpoint,
                summary.probe_median,
                summary.probe_stddev,
                summary.probe_max - summary.probe_min,
                summary.count,
            )

        fit_candidates = [result for result in fit_results if result.direction == "up"]
        if len(fit_candidates) < 2:
            msg = "Need at least two upward temperature points to fit expansion."
            raise RuntimeError(msg)

        fit_points = [(result.average_temp, result.probe_median) for result in fit_candidates]
        slope, intercept, r_squared = fit_linear_expansion(fit_points)

        self._append_summary_metrics(
            summary_dump_path,
            fit_results,
            model_name=model_name,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            applied=p.apply,
            unsafe_override_temp_limit=p.unsafe_override_temp_limit,
            sweep_mode=p.sweep_mode,
            probe_point=effective_probe_point,
            clean_above=p.clean_above,
            post_clean_dwell=p.post_clean_dwell,
        )

        logger.info(
            "Fitted expansion slope %.8f mm/°C (R²=%.5f)",
            slope,
            r_squared,
        )

        if p.apply:
            model = self._probe.get_model().config
            updated_model = replace(model, thermal_expansion_coefficient=slope)
            self._config.save_touch_model(updated_model)
            self._probe.load_model(updated_model.name)
            logger.info(
                "Saved expansion fit to touch model '%s'. Run SAVE_CONFIG to persist.",
                updated_model.name,
            )

    def _validate_homing(self) -> None:
        if not self._toolhead.is_homed("x") or not self._toolhead.is_homed("y") or not self._toolhead.is_homed("z"):
            msg = "Must home x, y, and z before touch expansion fitting"
            raise RuntimeError(msg)

    def _ensure_model_loaded(self, model_name: str | None) -> None:
        if model_name is not None:
            self._probe.load_model(model_name.lower())
            return
        if not self._probe.is_ready:
            msg = "No touch model is loaded. Pass MODEL=<name> or load one with CARTOGRAPHER_TOUCH_MODEL."
            raise RuntimeError(msg)

    def _validate_temperature_limits(self, temperatures: Sequence[int], unsafe_override: bool) -> None:
        if unsafe_override:
            logger.warning("UNSAFE_OVERRIDE_TEMP_LIMIT enabled: bypassing touch max temperature safety limit.")
            return

        max_allowed = self._config.touch.max_touch_temperature
        peak_temp = max(temperatures)
        if peak_temp > max_allowed:
            msg = (
                f"Requested temperature range reaches {peak_temp}°C, above touch max {max_allowed}°C. "
                "Set UNSAFE_OVERRIDE_TEMP_LIMIT=1 to force this test."
            )
            raise RuntimeError(msg)

    def _set_nozzle_temperature(self, setpoint: int, tolerance: float) -> None:
        self._gcode.run_gcode(f"M104 S{setpoint}")
        minimum = setpoint - tolerance
        maximum = setpoint + tolerance
        self._gcode.run_gcode(
            f"TEMPERATURE_WAIT SENSOR=extruder MINIMUM={minimum:.2f} MAXIMUM={maximum:.2f}"
        )

    def _wait_for_stability(
        self,
        *,
        setpoint: int,
        tolerance: float,
        stability_seconds: float,
        rate_limit: float,
        timeout: float,
    ) -> None:
        if stability_seconds <= 0:
            return

        window_size = max(2, int(np.ceil(stability_seconds)))
        window: list[float] = []
        start = time.monotonic()
        last_max_rate = 0.0
        last_window_min = float("nan")
        last_window_max = float("nan")

        while True:
            current = self._toolhead.get_extruder_temperature().current
            window.append(current)
            if len(window) > window_size:
                window.pop(0)

            if len(window) == window_size:
                last_window_min = min(window)
                last_window_max = max(window)
                in_band = max(abs(temp - setpoint) for temp in window) <= tolerance
                rates = [abs(window[idx] - window[idx - 1]) for idx in range(1, len(window))]
                max_rate = max(rates) if rates else 0.0
                last_max_rate = max_rate
                if in_band and max_rate <= rate_limit:
                    return

            if time.monotonic() - start > timeout:
                logger.warning(
                    "Temperature did not stabilize within %.0fs at %d°C. "
                    "Continuing run anyway. Last window: [%.2f..%.2f]°C, max drift %.3f°C/s "
                    "(limits: +/-%.2f°C, %.3f°C/s).",
                    timeout,
                    setpoint,
                    last_window_min,
                    last_window_max,
                    last_max_rate,
                    tolerance,
                    rate_limit,
                )
                return

            self._toolhead.dwell(1.0)

    def _move_to_probe_point(self, point: tuple[float, float]) -> None:
        if self._toolhead.get_position().z < self._config.touch.retract_distance:
            self._toolhead.move(z=self._config.touch.retract_distance, speed=self._config.general.lift_speed)
        self._toolhead.move(x=point[0], y=point[1], speed=self._config.general.travel_speed)
        self._toolhead.wait_moves()

    def _run_nozzle_clean(self) -> None:
        self._gcode.run_gcode("NOZZLE_CLEAN")

    def _go_to_wastebin(self) -> None:
        self._gcode.run_gcode("BOX_GO_TO_WASTEBIN")

    def _build_dump_paths(self, model_name: str) -> tuple[str, str]:
        os.makedirs(DEFAULT_DUMP_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_")
        raw_path = os.path.join(DEFAULT_DUMP_DIR, f"touch_expansion_raw_{safe_model_name}_{timestamp}.csv")
        summary_path = os.path.join(DEFAULT_DUMP_DIR, f"touch_expansion_summary_{safe_model_name}_{timestamp}.csv")
        return raw_path, summary_path

    def _init_raw_dump(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "setpoint_temp_c",
                    "sample_index",
                    "nozzle_temp_current_c",
                    "nozzle_temp_target_c",
                    "probe_z",
                    "direction",
                    "sequence_index",
                ]
            )

    def _append_raw_dump_row(self, path: str, row: tuple[int, int, float, float, float, str, int]) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _init_summary_progress_dump(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "setpoint_temp_c",
                    "avg_nozzle_temp_c",
                    "probe_median_z",
                    "probe_stddev_z",
                    "probe_min_z",
                    "probe_max_z",
                    "sample_count",
                    "direction",
                    "sequence_index",
                ]
            )

    def _append_summary_progress_row(self, path: str, result: TemperatureFitResult) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    result.setpoint,
                    f"{result.average_temp:.6f}",
                    f"{result.probe_median:.6f}",
                    f"{result.probe_stddev:.6f}",
                    f"{result.probe_min:.6f}",
                    f"{result.probe_max:.6f}",
                    result.count,
                    result.direction,
                    result.sequence_index,
                ]
            )

    def _append_summary_metrics(
        self,
        path: str,
        fit_results: Sequence[TemperatureFitResult],
        *,
        model_name: str,
        slope: float,
        intercept: float,
        r_squared: float,
        applied: bool,
        unsafe_override_temp_limit: bool,
        sweep_mode: SweepMode,
        probe_point: tuple[float, float] | None,
        clean_above: int,
        post_clean_dwell: float,
    ) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([])
            writer.writerow(["metric", "value"])
            writer.writerow(["model", model_name])
            writer.writerow(["slope_mm_per_c", f"{slope:.10f}"])
            writer.writerow(["intercept", f"{intercept:.10f}"])
            writer.writerow(["r_squared", f"{r_squared:.10f}"])
            writer.writerow(["sweep_mode", sweep_mode.value])
            writer.writerow(["setpoint_count", len(fit_results)])
            if probe_point is not None:
                writer.writerow(["probe_point", f"{probe_point[0]:.3f},{probe_point[1]:.3f}"])
            else:
                writer.writerow(["probe_point", ""])
            writer.writerow(["applied", str(applied)])
            writer.writerow(["unsafe_override_temp_limit", str(unsafe_override_temp_limit)])
            writer.writerow(["clean_above_temp_c", clean_above])
            writer.writerow(["post_clean_dwell_s", f"{post_clean_dwell:.3f}"])
