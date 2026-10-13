"""Broad, reproducible start/goal distributions for the arrival experiments.

The arena is a sampling region, not a set of walls. Endpoint clearance and a
generous geometry-based deadline do not certify joint collision-free feasibility.
No controller is involved in sampling, so PP and MPC can use identical seeds.
"""
from dataclasses import dataclass
import math

import numpy as np

from .config import EnvConfig


FAMILY_WEIGHTS = {
    "random": 0.45,
    "crossing": 0.20,
    "converging": 0.15,
    "unequal": 0.15,
    "parallel": 0.05,
}
FAMILIES = ("mixed", *FAMILY_WEIGHTS)


@dataclass
class Scenario:
    starts: np.ndarray
    goals: np.ndarray
    deadline: float
    family: str
    arena_half: float
    metadata: dict


class _RejectedGeometry(Exception):
    """Internal bounded rejection sampling signal."""


def _min_separation(points):
    if len(points) < 2:
        return None
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


class ScenarioSampler:
    """Sample independent missions from an explicitly configured distribution.

    Mixed draws use ``cfg.family_weights`` when supplied and FAMILY_WEIGHTS
    otherwise. A curriculum changes the configuration between stages; sampling
    never changes it implicitly. Each episode chooses a new arena size,
    geometry, vehicle identities, headings, initial speeds, and deadline slack.
    ``seed`` resets this sampler's private RNG only.
    """

    def __init__(self, cfg: EnvConfig, seed=None):
        self.cfg = cfg.validate()
        if cfg.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        weights = getattr(cfg, "family_weights", None)
        weights = FAMILY_WEIGHTS if weights is None else weights
        if not isinstance(weights, dict) or not weights:
            raise ValueError("family_weights must be a nonempty dictionary")
        if any(name not in FAMILY_WEIGHTS for name in weights):
            raise ValueError(f"family_weights keys must be in {tuple(FAMILY_WEIGHTS)}")
        try:
            values = np.asarray(list(weights.values()), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("family_weights must be finite nonnegative numbers") from exc
        total = float(values.sum())
        if (not np.isfinite(values).all() or np.any(values < 0)
                or not math.isfinite(total) or total <= 0):
            raise ValueError("family_weights must be finite, nonnegative, and have positive total")
        self.family_names = tuple(weights)
        self.family_probabilities = values / total
        self.start_gap = 2 * cfg.footprint_radius + cfg.collision_margin + 0.20
        # Cars parked anywhere within their goal discs should remain separated.
        self.goal_gap = 2 * cfg.footprint_radius + cfg.collision_margin + 2 * cfg.goal_tolerance
        self.seed(seed)

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.draw_index = 0

    def _points(self, candidate, gap, half, valid=None):
        points = []
        for i in range(self.cfg.n_cars):
            for _ in range(512):
                point = np.asarray(candidate(i), dtype=float)
                if np.any(np.abs(point) > half):
                    continue
                if points and np.any(np.linalg.norm(np.asarray(points) - point, axis=1) < gap):
                    continue
                if valid is not None and not valid(i, point):
                    continue
                points.append(point)
                break
            else:
                raise _RejectedGeometry
        return np.asarray(points)

    def _random(self, half):
        starts = self._points(lambda _: self.rng.uniform(-half, half, 2), self.start_gap, half)
        goals = self._points(
            lambda _: self.rng.uniform(-half, half, 2), self.goal_gap, half,
            lambda i, point: np.linalg.norm(point - starts[i]) >= self.cfg.min_trip,
        )
        return starts, goals, {}

    def _radial(self, half, converging):
        n = self.cfg.n_cars
        # Translation is sampled afresh, as is global orientation. Large teams
        # need a broader target cluster to retain the same physical clearance.
        hub = self.rng.uniform(-0.35 * half, 0.35 * half, 2)
        extent = float(np.min(half - np.abs(hub)))
        spacing = 2 * math.pi / n
        phase = self.rng.uniform(-math.pi, math.pi)
        angles = phase + np.arange(n) * spacing
        angles += self.rng.uniform(-0.04, 0.04, n) * spacing
        if converging:
            # The initial curriculum retains comparable trip lengths rather
            # than combining a timing lesson with very unequal journeys. These
            # old-style radii are translated/rotated each episode and shrink
            # to fit this sampled square. More cars require a wider target ring
            # for endpoint clearance, never a silently enlarged sampling map.
            scale = min(1.0, extent / 4.5)
            start_radius = self.rng.uniform(3.0, 4.5, n) * scale
            packing_radius = (self.goal_gap / (2 * math.sin(math.pi / n))
                              if n > 1 else 0.0)
            low = max(1.3 * scale, 1.12 * packing_radius)
            high = min(max(2.0 * scale, 1.22 * packing_radius), 0.85 * extent)
            if low >= high:
                raise _RejectedGeometry
            goal_radius = self.rng.uniform(low, high, n)
        else:
            start_radius = self.rng.uniform(0.83, 0.99, n) * extent
            goal_radius = self.rng.uniform(0.80, 0.99, n) * extent
        goal_angles = angles + math.pi + self.rng.uniform(-0.04, 0.04, n) * spacing
        starts = hub + start_radius[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        goals = hub + goal_radius[:, None] * np.column_stack((np.cos(goal_angles), np.sin(goal_angles)))
        return starts, goals, {"hub": hub.tolist(), "orientation": float(phase)}

    def _unequal(self, half):
        starts = self._points(lambda _: self.rng.uniform(-half, half, 2), self.start_gap, half)
        n = self.cfg.n_cars
        short = np.zeros(n, dtype=bool)
        short[self.rng.permutation(n)[:max(1, n // 2)]] = True
        if n == 1:
            short[0] = bool(self.rng.integers(2))
        # 0.9 * half only keeps the goal inside the arena; unequal_short_max is the knob.
        short_max = max(self.cfg.min_trip, min(self.cfg.unequal_short_max, 0.9 * half))
        long_min = max(self.cfg.min_trip, 0.85 * half)

        def candidate(i):
            if short[i]:
                angle = self.rng.uniform(-math.pi, math.pi)
                distance = self.rng.uniform(self.cfg.min_trip, short_max)
                return starts[i] + distance * np.array([math.cos(angle), math.sin(angle)])
            return self.rng.uniform(-half, half, 2)

        goals = self._points(
            candidate, self.goal_gap, half,
            lambda i, point: np.linalg.norm(point - starts[i]) >= (self.cfg.min_trip if short[i] else long_min),
        )
        return starts, goals, {}

    def _parallel(self, half):
        angle = self.rng.uniform(-math.pi, math.pi)
        distance = self.rng.uniform(0.6, 1.0) * half
        displacement = distance * np.array([math.cos(angle), math.sin(angle)])
        starts = self._points(
            lambda _: self.rng.uniform(-half, half, 2), self.goal_gap, half,
            lambda _, point: np.all(np.abs(point + displacement) <= half),
        )
        goals = starts + displacement
        return starts, goals, {"displacement": displacement.tolist(), "orientation": float(angle)}

    def _valid(self, starts, goals, half):
        start_sep, goal_sep = _min_separation(starts), _min_separation(goals)
        return (
            np.all(np.abs(starts) <= half)
            and np.all(np.abs(goals) <= half)
            and (start_sep is None or start_sep >= self.start_gap)
            and (goal_sep is None or goal_sep >= self.goal_gap)
            and np.all(np.linalg.norm(goals - starts, axis=1) >= self.cfg.min_trip)
        )

    def sample(self):
        cfg = self.cfg
        family = cfg.family
        if family == "mixed":
            family = str(self.rng.choice(self.family_names, p=self.family_probabilities))
        half = float(self.rng.uniform(cfg.arena_half_min, cfg.arena_half_max))
        for attempts in range(1, 257):
            try:
                if family in ("crossing", "converging"):
                    positions, goals, geometry = self._radial(half, family == "converging")
                else:
                    positions, goals, geometry = getattr(self, "_" + family)(half)
                if self._valid(positions, goals, half):
                    break
            except _RejectedGeometry:
                pass
        else:
            raise ValueError(
                f"Could not sample {family!r} with {cfg.n_cars} cars in +/-{half:.3f} m "
                f"after 256 attempts (start gap {self.start_gap:.3f}, goal gap {self.goal_gap:.3f}). "
                "Increase the arena size or reduce the number/size of cars."
            )

        # No vehicle index consistently receives a particular lane or trip type.
        order = self.rng.permutation(cfg.n_cars)
        positions, goals = positions[order], goals[order]
        trips = np.linalg.norm(goals - positions, axis=1)
        bearings = np.arctan2(goals[:, 1] - positions[:, 1], goals[:, 0] - positions[:, 0])
        heading_errors = self.rng.uniform(-cfg.heading_jitter, cfg.heading_jitter, cfg.n_cars)
        headings = (bearings + heading_errors + math.pi) % (2 * math.pi) - math.pi
        speeds = self.rng.uniform(0, cfg.initial_speed_max, cfg.n_cars)
        starts = np.column_stack((positions, headings, speeds, np.zeros(cfg.n_cars)))

        # Quintic rest-to-rest straight-line profile bounds on speed/acceleration.
        # Turning time and initially nonzero speed add heuristics, not a proof of
        # feasibility in the presence of other cars or nonholonomic constraints.
        translation_time = np.maximum(
            1.875 * trips / cfg.max_speed,
            np.sqrt((10 / math.sqrt(3)) * trips / min(cfg.max_accel, cfg.max_brake)),
        )
        nominal_yaw_rate = (0.5 * cfg.max_speed / cfg.wheelbase) * math.tan(cfg.max_steer)
        turning_time = np.abs(heading_errors) / max(nominal_yaw_rate, 1e-6)
        nominal_times = 1.02 * translation_time + turning_time + speeds / cfg.max_brake
        slack = float(self.rng.uniform(cfg.deadline_slack_min, cfg.deadline_slack_max))
        deadline = float(slack * nominal_times.max() + cfg.dwell_time)
        metadata = {
            "draw_index": self.draw_index,
            "sampling_attempts": attempts,
            "trip_lengths": trips.tolist(),
            "heading_errors": heading_errors.tolist(),
            "initial_speeds": speeds.tolist(),
            "car_permutation": order.tolist(),
            "deadline_slack": slack,
            "nominal_trip_times": nominal_times.tolist(),
            "deadline_rule": "quintic_speed_acceleration_plus_turning_heuristic",
            "deadline_is_feasibility_certificate": False,
            "arena_is_sampling_region_only": True,
            "minimum_start_separation": _min_separation(positions),
            "minimum_goal_separation": _min_separation(goals),
            **geometry,
        }
        self.draw_index += 1
        return Scenario(starts, goals, deadline, family, half, metadata)
