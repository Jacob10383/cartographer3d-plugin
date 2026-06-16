from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import chain
from math import isfinite
from typing import TYPE_CHECKING, final

from typing_extensions import override

from cartographer.interfaces.configuration import BedMeshConfig, MeshDirection, MeshPath, ScanConfig
from cartographer.interfaces.printer import (
    AxisTwistCompensation,
    Macro,
    MacroParams,
    Position,
    Sample,
    SupportsFallbackMacro,
    Toolhead,
)
from cartographer.lib.log import log_duration
from cartographer.macros.bed_mesh.helpers import (
    AdaptiveMeshCalculator,
    CoordinateTransformer,
    GridPointResult,
    MeshBounds,
    MeshGrid,
    Region,
    SampleProcessor,
    smooth_positions,
)
from cartographer.macros.bed_mesh.paths.alternating_snake import AlternatingSnakePathGenerator
from cartographer.macros.bed_mesh.paths.hilbert_path import HilbertPathGenerator
from cartographer.macros.bed_mesh.paths.random_path import RandomPathGenerator
from cartographer.macros.bed_mesh.paths.snake_path import SnakePathGenerator
from cartographer.macros.bed_mesh.paths.spiral_path import SpiralPathGenerator
from cartographer.macros.fields import config_ref, param
from cartographer.macros.utils import get_choice, get_float_tuple, get_int_tuple
from cartographer.probe.touch_mode import TOUCH_PLATE_MAX_X, TOUCH_PLATE_MAX_Y, TOUCH_PLATE_MIN_X, TOUCH_PLATE_MIN_Y

if TYPE_CHECKING:
    from cartographer.interfaces.configuration import Configuration
    from cartographer.interfaces.multiprocessing import TaskExecutor
    from cartographer.macros.bed_mesh.interfaces import BedMeshAdapter, PathGenerator, Point
    from cartographer.probe import Probe

logger = logging.getLogger(__name__)


def _parse_max_corner_radius(value: str, name: str) -> float | None:
    stripped = value.strip()
    if stripped.lower() == "auto":
        return None

    try:
        radius = float(stripped)
    except ValueError:
        msg = f"{name} must be 'auto' or a non-negative number, got {value!r}"
        raise ValueError(msg) from None

    if radius < 0:
        msg = f"{name} must be 'auto' or a non-negative number, got {radius}"
        raise ValueError(msg)

    return radius


def _get_max_corner_radius(params: MacroParams, config_default: float | None) -> float | None:
    value = params.get("MAX_CORNER_RADIUS", default=None)
    if value is None:
        return config_default
    return _parse_max_corner_radius(value, "MAX_CORNER_RADIUS")


@dataclass(frozen=True)
class BedMeshCalibrateConfiguration:
    mesh_min: tuple[float, float]
    mesh_max: tuple[float, float]
    probe_count: tuple[int, int]
    speed: float
    adaptive_margin: float
    zero_reference_position: Point
    faulty_regions: list[Region]

    runs: int
    direction: str
    height: float
    path: MeshPath
    max_corner_radius: float | None = None

    @staticmethod
    def from_config(config: Configuration):
        return BedMeshCalibrateConfiguration(
            mesh_min=config.bed_mesh.mesh_min,
            mesh_max=config.bed_mesh.mesh_max,
            probe_count=config.bed_mesh.probe_count,
            speed=config.bed_mesh.speed,
            adaptive_margin=config.bed_mesh.adaptive_margin,
            zero_reference_position=config.bed_mesh.zero_reference_position,
            runs=config.scan.mesh_runs,
            direction=config.scan.mesh_direction,
            height=config.scan.mesh_height,
            path=config.scan.mesh_path,
            max_corner_radius=config.scan.mesh_max_corner_radius,
            faulty_regions=list(map(lambda r: Region(r[0], r[1]), config.bed_mesh.faulty_regions)),
        )


_directions: list[str] = ["x", "y"]


PATH_GENERATOR_MAP = {
    MeshPath.SNAKE: SnakePathGenerator,
    MeshPath.ALTERNATING_SNAKE: AlternatingSnakePathGenerator,
    MeshPath.SPIRAL: SpiralPathGenerator,
    MeshPath.RANDOM: RandomPathGenerator,
    MeshPath.HILBERT: HilbertPathGenerator,
}


@dataclass(frozen=True)
class BedMeshScanAllParams:
    """Declarative parameter schema for BED_MESH_CALIBRATE.

    This dataclass is used only for unknown-param validation and docs generation.
    Actual parsing is handled by MeshScanParams.from_macro_params because several
    parameters (MESH_MIN, MESH_MAX, PROBE_COUNT) require comma-separated tuple parsing
    that the generic param() system does not support.
    """

    method: str = param("Calibration method (scan or touch)", default="scan")
    mesh_min: str | None = param("Minimum mesh coordinate (x,y)", default=None)
    mesh_max: str | None = param("Maximum mesh coordinate (x,y)", default=None)
    probe_count: str | None = param("Number of probe points (x,y)", default=None)
    adaptive: int = param("Enable adaptive meshing (0 or 1)", default=0)
    adaptive_margin: float = param(
        "Margin for adaptive mesh",
        default=config_ref(BedMeshConfig, "adaptive_margin"),
        min=0,
    )
    profile: str = param("Mesh profile name", default="default")
    direction: MeshDirection = param("Primary scan direction", default=config_ref(ScanConfig, "mesh_direction"))
    path: MeshPath = param("Scan path pattern", default=config_ref(ScanConfig, "mesh_path"))
    speed: float = param("Scan speed", default=config_ref(BedMeshConfig, "speed"), min=50)
    height: float = param("Scan height", default=config_ref(ScanConfig, "mesh_height"), min=0.5, max=5)
    runs: int = param("Number of scan passes", default=config_ref(ScanConfig, "mesh_runs"), min=1)
    iqr_reject: int = param("Enable per-cell IQR outlier rejection (0 or 1)", default=0)
    smooth: float = param("Gaussian smoothing sigma (0 to disable, requires scipy)", default=0.0, min=0.0, max=2.0)
    max_corner_radius: str | None = param(
        "Maximum corner radius (mm) for scan path arcs."
        " Use AUTO for automatic radius, 0 to disable smoothing arcs, or a positive number to cap auto radius.",
        default=None,
    )


@dataclass
class MeshScanParams:
    mesh_bounds: MeshBounds
    resolution: tuple[int, int]
    speed: float
    height: float
    runs: int
    adaptive: bool
    adaptive_margin: float
    profile: str | None
    iqr_reject: bool
    smooth: float
    path_generator: PathGenerator

    @classmethod
    def from_macro_params(
        cls, params: MacroParams, config: BedMeshCalibrateConfiguration, adapter: BedMeshAdapter
    ) -> MeshScanParams:
        """Create parameters from macro input and configuration."""
        base_bounds = MeshBounds(
            get_float_tuple(params, "MESH_MIN", default=config.mesh_min),
            get_float_tuple(params, "MESH_MAX", default=config.mesh_max),
        )
        base_resolution = get_int_tuple(params, "PROBE_COUNT", default=config.probe_count)

        adaptive = params.get_int("ADAPTIVE", default=0) != 0
        adaptive_margin = params.get_float("ADAPTIVE_MARGIN", config.adaptive_margin, minval=0)

        # Calculate actual bounds and resolution
        if adaptive:
            calculator = AdaptiveMeshCalculator(base_bounds, base_resolution)
            object_points = list(chain.from_iterable(adapter.get_objects()))
            mesh_bounds = calculator.calculate_adaptive_bounds(object_points, adaptive_margin)
            resolution = calculator.calculate_adaptive_resolution(mesh_bounds)
            profile = None  # Adaptive meshes don't use profiles
        else:
            mesh_bounds = base_bounds
            resolution = base_resolution
            profile = params.get("PROFILE", default="default")

        # Create path generator
        direction: str = get_choice(params, "DIRECTION", _directions, default=config.direction)
        path_type = get_choice(params, "PATH", default=config.path, choices=PATH_GENERATOR_MAP.keys())
        max_corner_radius = _get_max_corner_radius(params, config.max_corner_radius)
        path_generator = PATH_GENERATOR_MAP[path_type](direction, max_corner_radius)

        return cls(
            mesh_bounds=mesh_bounds,
            resolution=resolution,
            speed=params.get_float("SPEED", default=config.speed, minval=50),
            height=params.get_float("HEIGHT", default=config.height, minval=0.5, maxval=5),
            runs=params.get_int("RUNS", default=config.runs, minval=1),
            adaptive=adaptive,
            adaptive_margin=adaptive_margin,
            profile=profile,
            iqr_reject=params.get_int("IQR_REJECT", default=0, minval=0, maxval=1) != 0,
            smooth=params.get_float("SMOOTH", default=0.0, minval=0.0, maxval=2.0),
            path_generator=path_generator,
        )


@final
class BedMeshCalibrateMacro(Macro, SupportsFallbackMacro):
    description = "Gather samples across the bed to calibrate the bed mesh."

    def __init__(
        self,
        probe: Probe,
        toolhead: Toolhead,
        adapter: BedMeshAdapter,
        axis_twist_compensation: AxisTwistCompensation | None,
        task_executor: TaskExecutor,
        config: BedMeshCalibrateConfiguration,
    ):
        self.probe = probe
        self.toolhead = toolhead
        self.adapter = adapter
        self.task_executor = task_executor
        self.config = config
        self.coordinate_transformer = CoordinateTransformer(probe.scan.offset)
        self.axis_twist_compensation = axis_twist_compensation
        self._fallback: Macro | None = None

    @override
    def set_fallback_macro(self, macro: Macro) -> None:
        self._fallback = macro

    @override
    def run(self, params: MacroParams) -> None:
        """Main entry point for bed mesh calibration."""
        raw_method = params.get("METHOD", "scan")
        method = raw_method.lower()
        if method == "scan":
            return self._run_scan(params)
        if method == "touch":
            return self._run_touch(params)

        if self._fallback is None:
            msg = f"Bed mesh calibration method '{raw_method}' not supported"
            raise RuntimeError(msg)
        return self._fallback.run(params)

    def _run_scan(self, params: MacroParams) -> None:
        # Parse parameters and validate
        scan_params = MeshScanParams.from_macro_params(params, self.config, self.adapter)

        # Create mesh grid and processors
        grid = MeshGrid(
            scan_params.mesh_bounds.min_point,
            scan_params.mesh_bounds.max_point,
            scan_params.resolution[0],
            scan_params.resolution[1],
        )
        # Generate path and collect samples
        path = self._generate_path(grid, scan_params)
        self.adapter.clear_mesh()
        samples = self._collect_samples(path, scan_params)

        # Process samples and create mesh
        positions = self.task_executor.run(
            self._process_samples_to_positions,
            grid,
            samples,
            scan_params.height,
            scan_params.iqr_reject,
            scan_params.smooth,
        )
        positions = self._apply_zero_reference_height(positions, scan_params, grid)

        # Apply mesh to adapter
        self.adapter.apply_mesh(positions, scan_params.profile)

    def _run_touch(self, params: MacroParams) -> None:
        touch_params = MeshScanParams.from_macro_params(params, self.config, self.adapter)
        grid = self._create_touch_grid(touch_params)

        self.adapter.clear_mesh()
        positions = self._collect_touch_positions(grid.generate_points(), touch_params)
        positions = self.coordinate_transformer.apply_faulty_regions(positions, self.config.faulty_regions)

        if touch_params.smooth > 0:
            positions = smooth_positions(positions, touch_params.smooth)

        positions = self._apply_touch_zero_reference_height(positions, touch_params, grid)

        self.adapter.apply_mesh(positions, touch_params.profile)
        self._move_nozzle_to_point(self.config.zero_reference_position, touch_params.speed)
        self.toolhead.wait_moves()

    def _create_touch_grid(self, params: MeshScanParams) -> MeshGrid:
        ox, oy = self.probe.scan.offset.x, self.probe.scan.offset.y
        y_min, y_max = self.toolhead.get_axis_limits("y")

        min_x = TOUCH_PLATE_MIN_X - ox
        max_x = TOUCH_PLATE_MAX_X - ox
        min_y = max(float(y_min), TOUCH_PLATE_MIN_Y - oy)
        max_y = min(float(y_max), TOUCH_PLATE_MAX_Y - oy)

        if min_y > max_y:
            msg = (
                "Touch mesh has no valid Y travel range: "
                f"scanner plate Y=[{TOUCH_PLATE_MIN_Y:.2f}, {TOUCH_PLATE_MAX_Y:.2f}], "
                f"toolhead Y=[{y_min:.2f}, {y_max:.2f}], offset Y={oy:.2f}"
            )
            raise RuntimeError(msg)

        return MeshGrid(
            (min_x, min_y),
            (max_x, max_y),
            params.resolution[0],
            params.resolution[1],
        )

    def _apply_zero_reference_height(
        self, positions: list[Position], params: MeshScanParams, grid: MeshGrid
    ) -> list[Position]:
        zrp = self.config.zero_reference_position
        if grid.contains_point(zrp):
            return self.coordinate_transformer.normalize_to_zero_reference_point(positions, zero_ref=zrp)

        self._move_probe_to_point(zrp, params.speed)
        zero_measure = params.height - self.probe.scan.measure_distance()
        nx, ny = self.coordinate_transformer.probe_to_nozzle(zrp)
        if self.axis_twist_compensation:
            zero_measure += self.axis_twist_compensation.get_z_compensation_value(x=float(nx), y=float(ny))

        return self.coordinate_transformer.normalize_to_zero_reference_point(positions, zero_height=zero_measure)

    def _apply_touch_zero_reference_height(
        self, positions: list[Position], params: MeshScanParams, grid: MeshGrid
    ) -> list[Position]:
        zrp = self.config.zero_reference_position
        if grid.contains_point(zrp):
            return self.coordinate_transformer.normalize_to_zero_reference_point(positions, zero_ref=zrp)

        self._move_nozzle_to_point(zrp, params.speed)
        zero_measure = self.probe.perform_touch()
        return self.coordinate_transformer.normalize_to_zero_reference_point(positions, zero_height=zero_measure)

    def _generate_path(self, grid: MeshGrid, params: MeshScanParams) -> list[Point]:
        """Generate scanning path from grid points."""
        mesh_points = grid.generate_points()

        x_min, x_max = self.toolhead.get_axis_limits("x")
        y_min, y_max = self.toolhead.get_axis_limits("y")
        ox, oy = self.probe.scan.offset.x, self.probe.scan.offset.y

        return list(
            params.path_generator.generate_path(
                mesh_points,
                (x_min + max(0, ox), x_max + min(0, ox)),
                (y_min + max(0, oy), y_max + min(0, oy)),
            )
        )

    @log_duration("Collecting samples along the scanning path")
    def _collect_samples(self, path: list[Point], params: MeshScanParams) -> list[Sample]:
        """Collect samples by following the scanning path."""
        # Move to starting position
        self.toolhead.move(z=params.height, speed=5)
        self._move_probe_to_point(path[0], params.speed)
        self.toolhead.wait_moves()

        # Execute scan
        with self.probe.scan.start_session() as session:
            session.wait_for(lambda samples: len(samples) >= 10)

            for run_index in range(params.runs):
                sequence = path if run_index % 2 == 0 else reversed(path)
                for point in sequence:
                    self._move_probe_to_point(point, params.speed)

                self.toolhead.dwell(0.250)
                self.toolhead.wait_moves()

            # Wait for final samples
            move_time = self.toolhead.get_last_move_time()
            session.wait_for(lambda samples: samples[-1].time >= move_time)
            count = len(session.items)
            session.wait_for(lambda samples: len(samples) >= count + 10)

        samples = session.get_items()
        logger.debug("Collected %d samples across %d runs", len(samples), params.runs)
        return [self._transform_sample(s) for s in samples]

    @log_duration("Collecting touch samples across the mesh grid")
    def _collect_touch_positions(self, points: list[Point], params: MeshScanParams) -> list[Position]:
        """Collect one touch probe sequence at each mesh point."""
        self.toolhead.move(z=params.height, speed=5)
        positions: list[Position] = []
        for point in points:
            self._move_nozzle_to_point(point, params.speed)
            self.toolhead.wait_moves()
            positions.append(Position(x=float(point[0]), y=float(point[1]), z=self.probe.perform_touch()))

        logger.debug("Collected %d touch mesh samples", len(positions))
        return positions

    def _move_probe_to_point(self, point: Point, speed: float) -> None:
        """Move probe to specified point (converts to nozzle coordinates)."""
        x, y = self.coordinate_transformer.probe_to_nozzle(point)
        self.toolhead.move(x=float(x), y=float(y), speed=speed)

    def _move_nozzle_to_point(self, point: Point, speed: float) -> None:
        """Move nozzle directly to specified point."""
        self.toolhead.move(x=float(point[0]), y=float(point[1]), speed=speed)

    def _transform_sample(self, sample: Sample) -> Sample:
        """Transform sample to probe coordinates."""
        if sample.position is None:
            return sample

        probe_position = self.coordinate_transformer.nozzle_to_probe((sample.position.x, sample.position.y))
        return replace(
            sample, position=Position(x=float(probe_position[0]), y=float(probe_position[1]), z=sample.position.z)
        )

    @log_duration("Processing samples into final mesh positions")
    def _process_samples_to_positions(
        self, grid: MeshGrid, samples: list[Sample], height: float, iqr_reject: bool, smooth: float
    ) -> list[Position]:
        """Process samples into final mesh positions."""
        sample_processor = SampleProcessor(grid)

        logger.info("Processing %d samples into %dx%d grid...", len(samples), grid.x_resolution, grid.y_resolution)

        # Step 1: Compute heights
        heights = self.probe.scan.calculate_sample_distance_batch(samples)

        # Step 2: Bin samples to grid, optionally rejecting per-cell IQR outliers
        results = sample_processor.assign_samples_to_grid_batch(samples, heights, reject_outliers=iqr_reject)

        # Convert results to positions
        positions = self._results_to_positions(results, height)
        positions = self.coordinate_transformer.apply_faulty_regions(positions, self.config.faulty_regions)

        # Smooth only after faulty-region repair so bad cells do not bleed into neighbors.
        if smooth > 0:
            positions = smooth_positions(positions, smooth)

        return positions

    def _results_to_positions(self, results: list[GridPointResult], height: float) -> list[Position]:
        """Convert grid results to Position objects."""
        positions: list[Position] = []

        total_samples = sum(r.sample_count for r in results)
        invalid_points = [(r.point, r.sample_count) for r in results if not isfinite(r.z)]
        sparse_points = [(r.point, r.sample_count) for r in results if isfinite(r.z) and r.sample_count < 3]

        if invalid_points:
            invalid_list = ", ".join(f"({p[0]:.2f},{p[1]:.2f}) samples={n}" for p, n in invalid_points)
            lines = [
                f"Mesh scan failed: {len(invalid_points)}/{len(results)} grid points have no valid samples.",
                f"Total samples collected: {total_samples}.",
                f"Invalid grid points: {invalid_list}.",
            ]
            if sparse_points:
                sparse_list = ", ".join(f"({p[0]:.2f},{p[1]:.2f})={n}" for p, n in sparse_points)
                lines.append(f"Sparse grid points (<3 samples): {sparse_list}.")
            msg = " ".join(lines)
            logger.error(msg)
            raise RuntimeError(msg)

        if sparse_points:
            sparse_list = ", ".join(f"({p[0]:.2f},{p[1]:.2f})={n}" for p, n in sparse_points)
            logger.warning(
                "Mesh scan: %d/%d grid points have fewer than 3 samples: %s",
                len(sparse_points),
                len(results),
                sparse_list,
            )

        for result in results:
            rx, ry = result.point

            # Calculate compensated height
            z = height - result.z
            nx, ny = self.coordinate_transformer.probe_to_nozzle(result.point)
            if self.axis_twist_compensation:
                z += self.axis_twist_compensation.get_z_compensation_value(x=float(nx), y=float(ny))

            # Convert back to probe coordinates
            positions.append(Position(x=float(rx), y=float(ry), z=z))

        return positions
