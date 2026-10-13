"""Serializable configuration for corrected PP/MPC simultaneous-arrival training."""
from dataclasses import asdict, dataclass
import math
import numbers

DEFAULT_CONTROLLER = "pp"
EGO_DIM, NEIGHBOR_DIM, CRITIC_DIM = 10, 6, 16

EGO_FEATURES = (
    "goal_forward/10", "goal_lateral/10", "speed/max_speed", "steering/max_steer",
    "remaining_seconds/20", "deadline_seconds/20", "initial_distance/10", "arrived",
    "required_average_speed/max_speed", "progress_minus_elapsed_fraction",
)
CRITIC_FEATURES = (
    "x/10", "y/10", "sin_yaw", "cos_yaw", "speed/max_speed", "steering/max_steer",
    "goal_x/10", "goal_y/10", "remaining_seconds/20", "deadline_seconds/20",
    "initial_distance/10", "arrived", "distance_progress", "arrival_minus_deadline/20",
    "dwell_fraction", "arrival_bonus_paid",
)


@dataclass
class EnvConfig:
    controller: str = DEFAULT_CONTROLLER
    n_cars: int = 3
    max_neighbors: int = 4
    sensing_radius: float = 5.0
    policy_dt: float = 0.20
    control_dt: float = 0.05
    physics_dt: float = 0.01
    reference_horizon: float = 1.20
    mpc_horizon: float = 0.80
    wheelbase: float = 0.35
    max_speed: float = 2.0
    max_accel: float = 1.5
    max_brake: float = 2.0
    max_steer: float = math.radians(27)
    max_steer_rate: float = math.radians(90)
    # Conservative disc centered at the rear axle; unlike the old 0.28 m proxy,
    # it encloses the current F1TENTH chassis. Transfer still needs contact checks.
    # Collision geometry. "discs" places `footprint_discs` equal circles along the car's
    # own axis, sized to enclose it: this follows the shape instead of circumscribing the
    # whole car with one circle centred on the rear axle. Two discs need a 0.36 m car-car
    # gap where the single 0.50 m disc needed 1.00 m, while covering 1.6x the car's area
    # instead of 4.8x. "single" reproduces the legacy rear-axle disc of `footprint_radius`.
    collision_model: str = "discs"
    footprint_discs: int = 2
    car_length: float = 0.537            # measured from the MuJoCo F1TENTH body
    car_width: float = 0.245
    rear_axle_to_centre: float = 0.164   # the car centre is this far AHEAD of the rear axle
    footprint_radius: float = 0.50       # used only when collision_model == "single"
    # Separation constraints are imposed every `constraint_stride` horizon steps. The car
    # moves under 0.1 m per 0.05 s step, so every second step is ample and it keeps the
    # solver's constraint count (and therefore its runtime) manageable with multiple discs.
    constraint_stride: int = 2
    collision_margin: float = 0.08
    goal_tolerance: float = 0.25
    stop_speed: float = 0.08
    dwell_time: float = 0.20
    deadline_window: float = 0.60
    synchrony_window: float = 0.60
    timeout_margin: float = 3.0
    waypoint_forward_min: float = 0.25
    waypoint_forward_max: float = 2.0
    waypoint_lateral_max: float = 1.2
    waypoint_capture: float = 0.30
    pursuit_speed_gain: float = 4.0
    family: str = "mixed"
    family_weights: dict | None = None
    arena_half_min: float = 4.0
    arena_half_max: float = 7.0
    min_trip: float = 1.0
    # `unequal` draws the SHORT trips from [min_trip, unequal_short_max]. The default 2.0 m
    # against a 4-6 m long trip gives a ratio near 0.2, so the short car finishes in under a
    # second and must then loiter for ~14 s. Raise this to keep unequal trips without the
    # extreme precision-loitering case; set family_weights["unequal"]=0 to drop it entirely.
    unequal_short_max: float = 2.0
    heading_jitter: float = math.radians(60)
    initial_speed_max: float = 0.4
    deadline_slack_min: float = 1.5
    deadline_slack_max: float = 2.2
    schedule_weight: float = 0.5
    mpc_maxiter: int = 45

    def validate(self):
        if self.controller not in ("pp", "mpc"):
            raise ValueError("controller must be pp or mpc")
        if (not isinstance(self.n_cars, int) or not isinstance(self.max_neighbors, int)
                or self.n_cars < 1 or self.max_neighbors < 0):
            raise ValueError("n_cars must be positive; max_neighbors nonnegative")
        for name in ("policy_dt", "control_dt", "physics_dt", "reference_horizon",
                     "mpc_horizon", "wheelbase", "max_speed", "max_accel", "max_brake",
                     "max_steer", "max_steer_rate", "footprint_radius", "goal_tolerance",
                     "sensing_radius", "deadline_window", "synchrony_window", "dwell_time",
                     "timeout_margin", "waypoint_capture", "pursuit_speed_gain",
                     "waypoint_lateral_max", "waypoint_forward_min", "waypoint_forward_max"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for num, den in ((self.policy_dt, self.control_dt), (self.control_dt, self.physics_dt),
                         (self.reference_horizon, self.control_dt), (self.mpc_horizon, self.control_dt)):
            if abs(num / den - round(num / den)) > 1e-8:
                raise ValueError("clock intervals/horizons must divide evenly")
        if self.reference_horizon < self.mpc_horizon + self.policy_dt:
            raise ValueError("reference must cover MPC horizon plus one policy interval")
        if not 0 < self.arena_half_min <= self.arena_half_max:
            raise ValueError("invalid arena size range")
        if not 0 < self.min_trip < 2 * math.sqrt(2) * self.arena_half_min:
            raise ValueError("invalid min_trip")
        if not math.isfinite(self.unequal_short_max) or self.unequal_short_max < self.min_trip:
            raise ValueError("unequal_short_max must be finite and at least min_trip")
        if not 0 <= self.heading_jitter <= math.pi:
            raise ValueError("heading_jitter must be between 0 and pi radians")
        if not 0 <= self.initial_speed_max <= self.max_speed:
            raise ValueError("invalid initial speed range")
        if not 1 <= self.deadline_slack_min <= self.deadline_slack_max:
            raise ValueError("deadline slack must be >= 1")
        if self.family not in ("mixed", "random", "crossing", "converging", "unequal", "parallel"):
            raise ValueError("unknown scenario family")
        if self.family_weights is not None:
            if not isinstance(self.family_weights, dict) or not self.family_weights:
                raise ValueError("family_weights must be a nonempty dictionary or None")
            allowed = {"random", "crossing", "converging", "unequal", "parallel"}
            if set(self.family_weights) - allowed:
                raise ValueError("family_weights contains an unknown family")
            if any(not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0
                   for v in self.family_weights.values()):
                raise ValueError("family weights must be finite and nonnegative")
            total = sum(self.family_weights.values())
            if not math.isfinite(total) or total <= 0:
                raise ValueError("family weights must have finite positive total mass")
        if not 0 <= self.stop_speed <= self.max_speed:
            raise ValueError("invalid stop_speed")
        if self.collision_model not in ("discs", "single"):
            raise ValueError("collision_model must be 'discs' or 'single'")
        if not isinstance(self.footprint_discs, int) or self.footprint_discs < 1:
            raise ValueError("footprint_discs must be a positive integer")
        if not isinstance(self.constraint_stride, int) or self.constraint_stride < 1:
            raise ValueError("constraint_stride must be a positive integer")
        for name in ("car_length", "car_width"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.rear_axle_to_centre):
            raise ValueError("rear_axle_to_centre must be finite")
        if not math.isfinite(self.collision_margin) or self.collision_margin < 0:
            raise ValueError("collision margin must be nonnegative and finite")
        if not math.isfinite(self.schedule_weight) or self.schedule_weight < 0:
            raise ValueError("schedule_weight must be nonnegative and finite")
        if self.waypoint_forward_max <= self.waypoint_forward_min:
            raise ValueError("invalid waypoint forward range")
        if not isinstance(self.mpc_maxiter, int) or self.mpc_maxiter < 1:
            raise ValueError("mpc_maxiter must be a positive integer")
        return self

    def to_dict(self):
        return asdict(self)
