"""Alternative physics back-ends for zero-shot sim2sim transfer.

Every plant exposes the SAME five-state the policy and the MPC were trained on,
``[rear_axle_x, rear_axle_y, yaw, speed, steering_angle]``, and consumes the same
``[acceleration, steering_rate]`` control. Nothing above the plant changes: the
actor, the reference generator, the MPC and the reward are untouched.

    KinematicPlant  the nominal training model (identity; the reference row)
    DynamicPlant    single-track with Pacejka tyres, a friction ellipse, and
                    first-order steering/drive lag
    MujocoPlant     full rigid-body MuJoCo F1TENTH cars: torque-driven motor,
                    position-servo steering rack, real tyre contacts

Use with::

    env = ArrivalEnv(config, plant=MujocoPlant(config))

Caveats that matter for honest reporting. The dynamic plant's tyre and inertia
numbers are plausible, not identified from hardware. The MuJoCo car geometry
comes from the 11final scene and is close to, but not identical with, the
kinematic model's wheelbase. Neither establishes a real-vehicle guarantee.
"""
import math
import os
import sys

import numpy as np

from .model import bicycle_step

GRAVITY = 9.81


class KinematicPlant:
    """The nominal training model, wrapped in the plant interface."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, states):
        self.states = np.array(states, dtype=float)

    def step(self, controls, dt):
        self.states = np.stack([bicycle_step(s, u, dt, self.cfg)
                                for s, u in zip(self.states, controls)])
        return self.states.copy()


class DynamicPlant:
    """Single-track dynamics with tyre slip, load transfer limits and actuator lag.

    Internal state per car is ``[x_cg, y_cg, yaw, vx, vy, yaw_rate, steer,
    steer_command, drive_accel]``. Below ``blend_speed`` the lateral dynamics are
    blended toward the kinematic solution, because slip angles are singular at rest.
    """

    def __init__(self, cfg, friction=1.0, steer_lag=0.15, drive_lag=0.20,
                 mass=3.5, inertia=0.05, pacejka_b=5.0, pacejka_c=1.3,
                 drag=0.02, rolling=0.05):
        self.cfg = cfg
        self.mu, self.tau_steer, self.tau_drive = friction, steer_lag, drive_lag
        self.mass, self.inertia = mass, inertia
        self.lf = self.lr = cfg.wheelbase / 2
        self.pac_b, self.pac_c = pacejka_b, pacejka_c
        self.drag, self.rolling = drag, rolling
        self.blend_speed = 0.3

    def _lateral_force(self, slip, load):
        return self.mu * load * math.sin(self.pac_c * math.atan(self.pac_b * slip))

    def reset(self, states):
        states = np.asarray(states, dtype=float)
        self.raw = np.zeros((len(states), 9))
        for i, (x, y, yaw, speed, steer) in enumerate(states):
            self.raw[i, :2] = [x + self.lr * math.cos(yaw), y + self.lr * math.sin(yaw)]
            self.raw[i, 2:5] = yaw, speed, 0.0
            self.raw[i, 6:8] = steer, steer

    def _derivative(self, raw, accel_command, steer_rate):
        cfg = self.cfg
        x, y, yaw, vx, vy, yaw_rate, steer, steer_command, drive = raw
        wheelbase = self.lf + self.lr
        load_f = self.mass * GRAVITY * self.lr / wheelbase
        load_r = self.mass * GRAVITY * self.lf / wheelbase
        vx_safe = max(vx, self.blend_speed)
        slip_f = steer - math.atan2(vy + self.lf * yaw_rate, vx_safe)
        slip_r = -math.atan2(vy - self.lr * yaw_rate, vx_safe)
        fy_f = self._lateral_force(slip_f, load_f)
        fy_r = self._lateral_force(slip_r, load_r)
        # friction ellipse: lateral demand steals longitudinal capacity
        available = math.sqrt(max((self.mu * GRAVITY) ** 2 - (fy_r / self.mass) ** 2, 0.0))
        ax = (np.clip(drive, -available, available) - self.drag * vx * vx
              - (self.rolling if vx > 0.05 else 0.0))
        dvx = ax - fy_f * math.sin(steer) / self.mass + vy * yaw_rate
        dvy = (fy_f * math.cos(steer) + fy_r) / self.mass - vx * yaw_rate
        dyaw_rate = (self.lf * fy_f * math.cos(steer) - self.lr * fy_r) / self.inertia
        if vx < self.blend_speed:
            weight = vx / self.blend_speed
            kinematic_rate = vx * math.tan(steer) / wheelbase
            dyaw_rate = weight * dyaw_rate + (1 - weight) * (kinematic_rate - yaw_rate) / 0.05
            dvy = weight * dvy + (1 - weight) * (-vy / 0.05)
        return np.array([vx * math.cos(yaw) - vy * math.sin(yaw),
                         vx * math.sin(yaw) + vy * math.cos(yaw), yaw_rate,
                         dvx, dvy, dyaw_rate,
                         (steer_command - steer) / self.tau_steer, steer_rate,
                         (accel_command - drive) / self.tau_drive])

    def step(self, controls, dt, substeps=2):
        cfg = self.cfg
        h = dt / substeps
        for i, (accel, steer_rate) in enumerate(np.asarray(controls, dtype=float)):
            for _ in range(substeps):
                self.raw[i] += h * self._derivative(self.raw[i], accel, steer_rate)
                self.raw[i, 3] = np.clip(self.raw[i, 3], 0.0, cfg.max_speed)
                self.raw[i, 6] = np.clip(self.raw[i, 6], -cfg.max_steer, cfg.max_steer)
                self.raw[i, 7] = np.clip(self.raw[i, 7], -cfg.max_steer, cfg.max_steer)
        return self.states

    @property
    def states(self):
        out = np.zeros((len(self.raw), 5))
        for i, r in enumerate(self.raw):
            out[i] = [r[0] - self.lr * math.cos(r[2]), r[1] - self.lr * math.sin(r[2]),
                      r[2], r[3], r[6]]
        return out


class MujocoPlant:
    """MuJoCo F1TENTH cars driven through the same [accel, steer_rate] interface.

    The controller's acceleration is integrated into a commanded speed and fed to a
    proportional torque bridge on the drive tendon; the steering rate is integrated
    into a commanded angle and fed to the position servo on the steering rack. That
    is exactly how the physical car is driven, so the mismatch the policy sees here
    is real contact dynamics and real actuator response, not a re-parameterisation.
    """

    def __init__(self, cfg, torque_gain=30.0, force_limit=1.5 * GRAVITY,
                 friction_scale=1.0, render=False, scene_dir=None):
        import mujoco as mj
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(root, "11final"))
        import scene
        self.mj = mj
        self.cfg = cfg
        self.n = cfg.n_cars
        path = scene.generate(self.n, n_obst=0,
                              out_dir=scene_dir or os.path.join(root, "arrival_v2", "generated"))
        self.model = mj.MjModel.from_xml_path(path)
        self.model.pair_friction[:, :2] *= friction_scale
        self.data = mj.MjData(self.model)
        self.gain, self.force_limit = torque_gain, force_limit
        self.rear_offset = 0.16113          # body origin -> rear axle in the car XML
        self.qpos, self.qvel, self.drive, self.steer, self.steer_joint, self.mocap = [], [], [], [], [], []
        for i in range(self.n):
            joint = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"car_{i}_free")
            self.qpos.append(int(self.model.jnt_qposadr[joint]))
            self.qvel.append(int(self.model.jnt_dofadr[joint]))
            self.drive.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"drive_{i}"))
            self.steer.append(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"steer_{i}"))
            rack = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f"car_{i}_wfl_steer")
            self.steer_joint.append(int(self.model.jnt_qposadr[rack]))
            body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"target_{i}")
            self.mocap.append(int(self.model.body_mocapid[body]))
        self.viewer = None
        if render:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _car_state(self, i):
        pos, vel = self.qpos[i], self.qvel[i]
        x, y = self.data.qpos[pos], self.data.qpos[pos + 1]
        qw, qx, qy, qz = self.data.qpos[pos + 3:pos + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        vx, vy = self.data.qvel[vel], self.data.qvel[vel + 1]
        forward = vx * math.cos(yaw) + vy * math.sin(yaw)
        return np.array([x - self.rear_offset * math.cos(yaw),
                         y - self.rear_offset * math.sin(yaw),
                         yaw, forward, self.data.qpos[self.steer_joint[i]]])

    def reset(self, states):
        states = np.asarray(states, dtype=float)
        if len(states) != self.n:
            raise ValueError(f"MujocoPlant was built for {self.n} cars, got {len(states)}")
        self.mj.mj_resetData(self.model, self.data)
        self.speed_command = states[:, 3].copy()
        self.steer_command = states[:, 4].copy()
        for i, (x, y, yaw, speed, steer) in enumerate(states):
            pos, vel = self.qpos[i], self.qvel[i]
            self.data.qpos[pos:pos + 2] = [x + self.rear_offset * math.cos(yaw),
                                           y + self.rear_offset * math.sin(yaw)]
            self.data.qpos[pos + 2] = 0.052388
            self.data.qpos[pos + 3:pos + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            self.data.qvel[vel:vel + 6] = 0
        self.mj.mj_forward(self.model, self.data)
        self.substeps = max(1, round(self.cfg.physics_dt / self.model.opt.timestep))

    def set_goals(self, goals):
        """Draw the targets in the MuJoCo window (visual only)."""
        for i, goal in enumerate(goals):
            self.data.mocap_pos[self.mocap[i]][:2] = goal

    def step(self, controls, dt):
        cfg = self.cfg
        for i, (accel, steer_rate) in enumerate(np.asarray(controls, dtype=float)):
            self.speed_command[i] = np.clip(self.speed_command[i] + accel * dt, 0.0, cfg.max_speed)
            self.steer_command[i] = np.clip(self.steer_command[i] + steer_rate * dt,
                                            -cfg.max_steer, cfg.max_steer)
        for _ in range(self.substeps):
            for i in range(self.n):
                speed = self._car_state(i)[3]
                self.data.ctrl[self.drive[i]] = float(np.clip(
                    self.gain * (self.speed_command[i] - speed), -self.force_limit, self.force_limit))
                self.data.ctrl[self.steer[i]] = float(self.steer_command[i])
            self.mj.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        return self.states

    @property
    def states(self):
        return np.stack([self._car_state(i) for i in range(self.n)])

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_plant(name, cfg, **kwargs):
    """Factory used by the CLI flags: 'kinematic' | 'dynamic' | 'mujoco'."""
    if name in (None, "kinematic"):
        return None                     # None means "use the env's built-in nominal model"
    if name == "dynamic":
        return DynamicPlant(cfg, **kwargs)
    if name == "mujoco":
        return MujocoPlant(cfg, **kwargs)
    raise ValueError(f"unknown plant {name!r}")
