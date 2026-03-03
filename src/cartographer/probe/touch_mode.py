from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from cartographer.interfaces.errors import ProbeTriggerError
from cartographer.interfaces.printer import (
    Endstop,
    HomingState,
    Mcu,
    Position,
    ProbeMode,
    Toolhead,
)
from cartographer.probe.touch_model import TouchModelSelectorMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cartographer.interfaces.configuration import (
        Configuration,
        TouchModelConfiguration,
    )
    from cartographer.probe.scan_mode import ScanMode

logger = logging.getLogger(__name__)


TOUCH_ACCEL = 100
MAX_TOUCH_TEMPERATURE_EPSILON = 2


@dataclass(frozen=True)
class TouchModeConfiguration:
    samples: int
    max_samples: int
    max_window: int

    x_offset: float
    y_offset: float
    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]
    max_touch_temperature: int
    lift_speed: float
    travel_speed: float

    retract_distance: float
    models: dict[str, TouchModelConfiguration]
    sample_range: float

    @staticmethod
    def from_config(config: Configuration):
        return TouchModeConfiguration(
            samples=config.touch.samples,
            max_samples=config.touch.max_samples,
            max_window=config.touch.samples + config.touch.max_noisy_samples,
            models=config.touch.models,
            x_offset=config.general.x_offset,
            y_offset=config.general.y_offset,
            mesh_min=config.bed_mesh.mesh_min,
            mesh_max=config.bed_mesh.mesh_max,
            max_touch_temperature=config.touch.max_touch_temperature,
            lift_speed=config.general.lift_speed,
            travel_speed=config.general.travel_speed,
            retract_distance=config.touch.retract_distance,
            sample_range=config.touch.sample_range,
        )


class TouchError(RuntimeError):
    pass


def run_probe_sequence(
    probe_fn: Callable[[], float],
    *,
    samples: int,
    max_samples: int,
    max_window: int,
    sample_range: float,
    log_start: bool = True,
) -> float:
    """
    Collect touch samples and find a consistent subset.

    Uses a sliding window of the most recent max_window samples
    to prevent cherry-picking good samples from across a noisy
    sequence. Exits early as soon as a valid subset is found.
    """
    collected: list[float] = []

    if log_start:
        logger.debug(
            "Starting touch sequence for %d samples within %d touches (window=%d)...",
            samples,
            max_samples,
            max_window,
        )

    for i in range(max_samples):
        trigger_pos = probe_fn()
        collected.append(trigger_pos)
        logger.debug("Touch %d: %.4f", i + 1, trigger_pos)

        if len(collected) < samples:
            continue

        window = collected[-max_window:]
        best = find_best_subset(window, samples)
        if best is None:
            continue

        best_range = compute_range(best)
        if best_range > sample_range:
            continue

        median = float(np.median(best))

        logger.debug(
            "Acceptable samples found: (%s)\nrange %.4f (limit %.4f), min %.4f, max %.4f,\nmean %.4f, median %.4f",
            ", ".join(f"{s:.4f}" for s in best),
            best_range,
            sample_range,
            min(best),
            max(best),
            float(np.mean(best)),
            median,
        )

        return median

    msg = (
        f"Unable to find {samples:d} samples within "
        f"{sample_range:.3f}mm in a window of {max_window:d} "
        f"after {max_samples:d} touches"
    )
    raise TouchError(msg)


@dataclass(frozen=True)
class TouchBoundaries:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def is_within(self, *, x: float, y: float) -> bool:
        epsilon = 0.01
        in_x_bounds = (self.min_x - epsilon) <= x <= (self.max_x + epsilon)
        in_y_bounds = (self.min_y - epsilon) <= y <= (self.max_y + epsilon)
        return in_x_bounds and in_y_bounds

    @staticmethod
    def from_config(config: TouchModeConfiguration) -> TouchBoundaries:
        mesh_min_x, mesh_min_y = config.mesh_min
        mesh_max_x, mesh_max_y = config.mesh_max
        x_offset = config.x_offset
        y_offset = config.y_offset

        min_x = mesh_min_x - min(x_offset, 0)
        min_y = mesh_min_y - min(y_offset, 0)
        max_x = mesh_max_x - max(x_offset, 0)
        max_y = mesh_max_y - max(y_offset, 0)

        return TouchBoundaries(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )


def compute_range(samples: Sequence[float]) -> float:
    """Compute the range (max - min) of samples."""
    if len(samples) < 2:
        return float("inf")
    return max(samples) - min(samples)


def find_best_subset(
    samples: Sequence[float],
    size: int,
) -> Sequence[float] | None:
    """Find the subset of samples with the smallest range."""
    result = heapq.nsmallest(
        1,
        combinations(samples, size),
        key=compute_range,
    )
    return result[0] if result else None


class TouchMode(TouchModelSelectorMixin, ProbeMode, Endstop):
    """Implementation for Survey Touch."""

    @property
    def retract_distance(self) -> float:
        return self._config.retract_distance

    def is_pre_touch_triggered(self) -> bool | None:
        if self._scan_mode is None or not self._scan_mode.is_ready:
            return None
        time = self._toolhead.get_last_move_time()
        return self._scan_mode.query_is_triggered(time)

    @property
    @override
    def offset(self) -> Position:
        return Position(0.0, 0.0, 0.0)

    @property
    @override
    def is_ready(self) -> bool:
        return self.has_model()

    @property
    @override
    def last_homing_time(self) -> float:
        return self._last_homing_time

    def __init__(
        self,
        mcu: Mcu,
        toolhead: Toolhead,
        config: TouchModeConfiguration,
        scan_mode: ScanMode | None = None,
    ) -> None:
        super().__init__(config.models)
        self._last_homing_time: float = 0.0
        self._toolhead: Toolhead = toolhead
        self._mcu: Mcu = mcu
        self._config: TouchModeConfiguration = config
        self._scan_mode: ScanMode | None = scan_mode

        self.boundaries: TouchBoundaries = TouchBoundaries.from_config(config)
        self.last_z_result: float | None = None
        self._threshold_override: int | None = None
        self._ignore_temp_limit: bool = False

    @override
    def get_status(self, eventtime: float) -> dict[str, object]:
        return {
            "current_model": (self.get_model().name if self.has_model() else "none"),
            "models": ", ".join(self._config.models.keys()),
            "last_z_result": round(self.last_z_result, 6) if self.last_z_result is not None else None,
        }

    @override
    def perform_probe(
        self,
        threshold_override: int | None = None,
        speed_override: float | None = None,
        *,
        ignore_temp_limit: bool = False,
        log_sequence_start: bool = True,
        log_touch_settings: bool = True,
        use_scan_approach: bool = True,
    ) -> float:
        if not self._toolhead.is_homed("z"):
            msg = "Z axis must be homed before probing"
            raise RuntimeError(msg)

        fast_start = self._pre_touch_approach_with_scan() if use_scan_approach else False

        if not fast_start:
            if self._toolhead.get_position().z < self._config.retract_distance:
                self._toolhead.move(z=self._config.retract_distance, speed=self._config.lift_speed)
            self._toolhead.wait_moves()

        self.last_z_result = self._run_probe(
            threshold_override,
            speed_override,
            ignore_temp_limit=ignore_temp_limit,
            log_sequence_start=log_sequence_start,
            log_touch_settings=log_touch_settings,
            fast_start=fast_start,
        )
        return self.last_z_result

    def _pre_touch_approach_with_scan(self) -> bool:
        """Use scan trigger for a coarse Z approach before touch probing."""
        if self._scan_mode is None or not self._scan_mode.is_ready:
            return False

        print_time = self._toolhead.get_last_move_time()
        try:
            if self._scan_mode.query_is_triggered(print_time):
                return True
            self._toolhead.z_probing_move(
                self._scan_mode,
                speed=self._scan_mode.probe_speed,
            )
            self._toolhead.wait_moves()
            return True
        except RuntimeError as err:
            logger.debug("Scan pre-approach skipped: %s", err)
            return False

    def _run_probe(
        self,
        threshold_override: int | None = None,
        speed_override: float | None = None,
        *,
        ignore_temp_limit: bool = False,
        log_sequence_start: bool = True,
        log_touch_settings: bool = True,
        fast_start: bool = False,
    ) -> float:
        if log_sequence_start:
            logger.info(
                "Starting touch sequence for %d samples within %d touches (window=%d)...",
                self._config.samples,
                self._config.max_samples,
                self._config.max_window,
            )

        if log_touch_settings:
            model = self.get_model()
            threshold = threshold_override if threshold_override is not None else model.threshold
            speed = speed_override if speed_override is not None else model.speed
            logger.info(
                "Touch settings: threshold %d%s, speed %.1f mm/s%s, z_offset %.3f mm",
                threshold,
                " (override)" if threshold_override is not None else " (model)",
                speed,
                " (override)" if speed_override is not None else " (model)",
                model.z_offset,
            )

        collected: list[float] = []

        def _probe_sample() -> float:
            # Retry within a single touch slot when we detect a phantom trigger.
            while True:
                trigger_pos = self._perform_single_probe(
                    threshold_override,
                    speed_override,
                    allow_fast_start=(fast_start and not collected),
                    ignore_temp_limit=ignore_temp_limit,
                )

                # If a trigger lands exactly one retract distance from the previous sample,
                # it is likely from the retract move and should be ignored.
                if collected:
                    last_sample = collected[-1]
                    delta = abs(trigger_pos - last_sample)
                    if abs(delta - self._config.retract_distance) < 0.01:
                        logger.warning(
                            "!! Phantom trigger ignored: %.4f (exactly +%.1fmm from previous %.4f)",
                            trigger_pos,
                            self._config.retract_distance,
                            last_sample,
                        )
                        continue

                collected.append(trigger_pos)
                return trigger_pos

        median = run_probe_sequence(
            _probe_sample,
            samples=self._config.samples,
            max_samples=self._config.max_samples,
            max_window=self._config.max_window,
            sample_range=self._config.sample_range,
            log_start=False,
        )
        return median
    def _perform_single_probe(
        self,
        threshold_override: int | None = None,
        speed_override: float | None = None,
        *,
        allow_fast_start: bool = False,
        ignore_temp_limit: bool = False,
    ) -> float:
        model = self.get_model()
        self._threshold_override = threshold_override
        self._ignore_temp_limit = ignore_temp_limit
        probe_speed = speed_override if speed_override is not None else model.speed

        if not allow_fast_start and self._toolhead.get_position().z < self._config.retract_distance:
            self._toolhead.move(z=self._config.retract_distance, speed=self._config.lift_speed)
            self._toolhead.wait_moves()
        elif not allow_fast_start:
            self._toolhead.wait_moves()

        max_accel = self._toolhead.get_max_accel()
        self._toolhead.set_max_accel(TOUCH_ACCEL)

        # Wait for a single new sample
        # This seems to help avoid triggering prior to move
        time = self._toolhead.get_last_move_time()
        with self._mcu.start_session(lambda sample: sample.time >= time):
            pass

        try:
            try:
                trigger_pos = self._toolhead.z_probing_move(self, speed=probe_speed)
            except ProbeTriggerError:
                if not allow_fast_start:
                    raise
                logger.debug("Fast-start touch probe triggered before movement; retrying with safety retract.")
                self._toolhead.move(z=self._config.retract_distance, speed=self._config.lift_speed)
                self._toolhead.wait_moves()
                trigger_pos = self._toolhead.z_probing_move(self, speed=probe_speed)
        finally:
            self._toolhead.set_max_accel(max_accel)
            self._threshold_override = None
            self._ignore_temp_limit = False

        pos = self._toolhead.get_position()
        self._toolhead.move(
            z=max(pos.z + self._config.retract_distance, self._config.retract_distance),
            speed=self._config.lift_speed,
        )

        return trigger_pos - model.z_offset

    @override
    def home_start(self, print_time: float) -> object:
        model = self.get_model()
        # Use threshold override if set, otherwise use model threshold
        threshold = getattr(self, '_threshold_override', None) or model.threshold
        if threshold <= 0:
            msg = "Threshold must be positive"
            raise RuntimeError(msg)

        pos = self._toolhead.get_position()
        if not self.boundaries.is_within(x=pos.x, y=pos.y):
            msg = (
                f"Position ({pos.x:.2f}, {pos.y:.2f}) is outside touch boundaries. "
                f"Valid range: X=[{self.boundaries.min_x:.2f}, {self.boundaries.max_x:.2f}], "
                f"Y=[{self.boundaries.min_y:.2f}, {self.boundaries.max_y:.2f}]"
            )
            raise RuntimeError(msg)

        nozzle_temperature = max(self._toolhead.get_extruder_temperature())
        max_temp = self._config.max_touch_temperature
        if not self._ignore_temp_limit:
            if nozzle_temperature > max_temp + MAX_TOUCH_TEMPERATURE_EPSILON:
                msg = f"Nozzle temperature must be below {max_temp:d}C, was {nozzle_temperature:.1f}C"
                raise RuntimeError(msg)
        return self._mcu.start_homing_touch(print_time, threshold)

    @override
    def on_home_end(self, homing_state: HomingState) -> None:
        if not homing_state.is_homing_z():
            return
        self._last_homing_time = self._toolhead.get_last_move_time()

    @override
    def home_wait(self, home_end_time: float) -> float:
        return self._mcu.stop_homing(home_end_time)

    @override
    def query_is_triggered(self, print_time: float) -> bool:
        return False

    @override
    def get_endstop_position(self) -> float:
        return self.offset.z
