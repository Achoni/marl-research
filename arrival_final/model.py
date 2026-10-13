"""Nominal, no-slip bicycle dynamics used by training and reference prediction.

State is ``[rear_axle_x, rear_axle_y, yaw, speed, steering_angle]``; input is
``[acceleration, steering_rate]``. All quantities use SI units and radians.
The state update clips speed/steering and uses midpoint integration for pose.
"""
import numpy as np


def wrap_angle(angle):
    """Wrap a scalar or array of angles into [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def bound_control(control, cfg):
    """Apply the shared acceleration and steering-rate actuator limits."""
    return np.clip(np.asarray(control, dtype=float),
                   [-cfg.max_brake, -cfg.max_steer_rate],
                   [cfg.max_accel, cfg.max_steer_rate])


def bicycle_step(state, control, dt, cfg):
    """Advance the nominal bicycle without reversing or instantaneous steering.

    This same integrator is used at the environment's physics interval and at
    the reference interval; subdividing a curved interval can therefore give
    slightly different numerical integration results.
    """
    state = np.asarray(state, dtype=float)
    accel, steer_rate = bound_control(control, cfg)
    speed = np.clip(state[3] + dt * accel, 0.0, cfg.max_speed)
    steer = np.clip(state[4] + dt * steer_rate, -cfg.max_steer, cfg.max_steer)
    speed_mid = 0.5 * (state[3] + speed)
    steer_mid = 0.5 * (state[4] + steer)
    yaw_change = dt * speed_mid * np.tan(steer_mid) / cfg.wheelbase
    yaw_mid = state[2] + 0.5 * yaw_change
    return np.array([
        state[0] + dt * speed_mid * np.cos(yaw_mid),
        state[1] + dt * speed_mid * np.sin(yaw_mid),
        wrap_angle(state[2] + yaw_change), speed, steer,
    ])


def body_discs(cfg):
    """Circles that cover the car, expressed as (offsets along the car axis, radius).

    Offsets are metres FORWARD of the rear axle, so a disc's world centre is
    ``state[:2] + offset * [cos(yaw), sin(yaw)]``. Every disc has the same radius,
    the smallest that covers its slice of the body including the full half-width.

    Splitting an enclosing rectangle into equal longitudinal slices gives
    radius sqrt((length/(2*count))**2 + (width/2)**2) for each slice.
    """
    if getattr(cfg, "collision_model", "single") == "single":
        return np.zeros(1), cfg.footprint_radius
    count = cfg.footprint_discs
    length, half_width = cfg.car_length, 0.5 * cfg.car_width
    radius = float(np.hypot(length / (2 * count), half_width))
    rear = cfg.rear_axle_to_centre - 0.5 * length          # tail, relative to the rear axle
    offsets = np.array([rear + (i + 0.5) * length / count for i in range(count)])
    return offsets, radius


def disc_centres(state, cfg):
    """World positions of one car's collision discs, shape (n_discs, 2)."""
    offsets, _ = body_discs(cfg)
    heading = np.array([np.cos(state[2]), np.sin(state[2])])
    return np.asarray(state[:2])[None, :] + offsets[:, None] * heading[None, :]


def car_separation(state_a, state_b, cfg):
    """Smallest gap between any disc of car A and any disc of car B, minus their radii.

    Zero means the two footprints touch; negative means they overlap.
    """
    _, radius = body_discs(cfg)
    a, b = disc_centres(state_a, cfg), disc_centres(state_b, cfg)
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min() - 2 * radius)
