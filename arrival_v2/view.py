"""Watch a trained PP or MPC policy in an interactive window (no training updates).

Run ``python3 -m arrival_v2.view`` from the repository, or ``python3 view.py``
inside this folder. New scenarios play automatically; --hold keeps the final
frame. Space pauses, N advances the seed, R replays, arrows change speed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import secrets
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

import numpy as np
import torch

from .config import EnvConfig
from .env import ArrivalEnv
from .mappo import MAPPOAgent
from .scenarios import FAMILIES


def resolve_checkpoint(path=None):
    """Prefer the newest best.pt in this standalone folder's runs directory."""
    if path is not None:
        checkpoint = Path(path).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint}")
        return checkpoint
    runs = Path(__file__).resolve().parent / "runs"
    candidates = list(runs.glob("*/best.pt")) or list(runs.glob("*/latest.pt"))
    if not candidates:
        raise ValueError("No checkpoint found in runs/. Supply --checkpoint /path/to/best.pt")
    return max(candidates, key=lambda item: item.stat().st_mtime).resolve()


def load_policy(checkpoint, device="cpu", cars=None, neighbors=None, family=None,
                controller=None, distribution="target", collision_model=None, footprint_discs=None):
    agent, metadata = MAPPOAgent.from_checkpoint(checkpoint, device=device)
    saved_config = metadata.get("active_env_config", metadata["env_config"]) if distribution == "training" else metadata["env_config"]
    config = EnvConfig(**saved_config)
    overrides = {key: value for key, value in {
        "n_cars": cars, "max_neighbors": neighbors, "family": family, "controller": controller,
        "collision_model": collision_model, "footprint_discs": footprint_discs,
    }.items() if value is not None}
    config = replace(config, **overrides).validate()
    agent.actor.eval()
    return agent, config, metadata


class LiveViewer:
    """Draw actual environment states; display updates never change the plant."""

    def __init__(self, agent, config, checkpoint, trained_steps, seed, speed=1.0, hold=False):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon
        from matplotlib.widgets import Button

        self.plt = plt
        self.agent = agent
        self.env = ArrivalEnv(config)
        self.speed, self.hold = speed, hold
        self.paused = False
        self.elapsed = self.end_elapsed = 0.0
        self.last_wall = time.perf_counter()
        self.figure = plt.figure(figsize=(12, max(7.5, 4 + 0.48 * config.n_cars)))
        self.figure.canvas.manager.set_window_title(f"Arrival v2 ({config.controller.upper()}) — {checkpoint.name}")
        label = "pure pursuit" if config.controller == "pp" else "pure pursuit reference + MPC"
        self.figure.suptitle(f"Learned policy + {label}", x=0.06, ha="left", fontsize=17)
        self.ax = self.figure.add_axes([0.07, 0.15, 0.57, 0.73])
        self.panel = self.figure.add_axes([0.69, 0.15, 0.30, 0.73])
        self.panel.axis("off")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(alpha=0.18)
        self.figure.text(0.06, 0.915,
                         f"{checkpoint.parent.name}/{checkpoint.name}  |  "
                         f"checkpoint step {trained_steps:,}" if isinstance(trained_steps, int)
                         else f"{checkpoint.parent.name}/{checkpoint.name}", fontsize=10)
        self.header = self.panel.text(0, 1, "", va="top", fontsize=11, linespacing=1.5)
        self.summary = self.panel.text(0, 0.05, "", va="bottom", fontsize=10, linespacing=1.5)
        palette = plt.get_cmap("tab10" if config.n_cars <= 10 else "tab20")
        self.colors = [palette(i % palette.N) for i in range(config.n_cars)]
        self.trails, self.bodies, self.footprints, self.labels = [], [], [], []
        self.goal_markers, self.goal_rings, self.goal_labels, self.car_info = [], [], [], []
        self.headings = []
        for i, color in enumerate(self.colors):
            self.trails.append(self.ax.plot([], [], color=color, lw=1.6, alpha=0.7)[0])
            self.goal_markers.append(self.ax.plot([], [], marker="x", color=color,
                                                  markersize=9, mew=2, linestyle="none")[0])
            ring = Circle((0, 0), config.goal_tolerance, fill=False, edgecolor=color, lw=1.5)
            footprint = Circle((0, 0), config.footprint_radius, fill=False,
                               edgecolor=color, linestyle=":", alpha=0.6)
            body = Polygon(np.zeros((4, 2)), closed=True, facecolor=color, edgecolor="white")
            for patch in (ring, footprint, body):
                self.ax.add_patch(patch)
            self.goal_rings.append(ring)
            self.footprints.append(footprint)
            self.bodies.append(body)
            self.headings.append(self.ax.plot([], [], color="#222222", lw=1.5)[0])
            self.labels.append(self.ax.text(0, 0, str(i + 1), color=color, fontsize=9))
            self.goal_labels.append(self.ax.text(0, 0, f"G{i + 1}", color=color, fontsize=9))
            self.car_info.append(self.panel.text(0, 0.69 - i * 0.43 / max(config.n_cars, 1),
                                                 "", va="top", color=color, fontsize=10))
        self.buttons = []  # Retain widgets so their callbacks remain live.
        for label, x, callback in (("Pause / play", 0.08, self.toggle_pause),
                                   ("Replay", 0.25, self.replay),
                                   ("Next scenario", 0.42, self.next_scenario)):
            button = Button(self.figure.add_axes([x, 0.065, 0.15, 0.045]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.figure.text(0.07, 0.025,
                         "Space: pause   N: next   R: replay   →: step   ↑/↓: speed   Esc: close",
                         fontsize=10)
        self.figure.text(0.69, 0.09, "× goal   ··· collision footprint\nLines show driven trajectories", fontsize=9)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("close_event", self.close)
        self.timer = self.figure.canvas.new_timer(interval=30)
        self.timer.add_callback(self.on_timer)
        self.reset(seed)

    def reset(self, seed):
        self.seed = seed
        self.observation = self.env.reset(seed=seed)
        self.info = self.env.info()
        self.history = [self.env.states[:, :2].copy()]
        self.elapsed = self.end_elapsed = 0.0
        self.last_wall = time.perf_counter()
        self.extent = max(float(self.env.scenario.arena_half),
                          float(np.max(np.abs(self.env.goals))),
                          float(np.max(np.abs(self.env.states[:, :2])))) + 0.8
        self.ax.set_xlim(-self.extent, self.extent)
        self.ax.set_ylim(-self.extent, self.extent)
        for i, goal in enumerate(self.env.goals):
            self.goal_markers[i].set_data([goal[0]], [goal[1]])
            self.goal_rings[i].center = goal
            self.goal_labels[i].set_position(goal + [0.15, 0.20])
        print(f"Scenario seed={seed} family={self.env.family} deadline={self.env.deadline:.2f}s", flush=True)
        self.draw()

    def advance(self):
        if self.env.done:
            return
        actions = self.agent.predict(self.observation, deterministic=True)
        self.observation, _, _, self.info = self.env.step(actions)
        self.history.append(self.env.states[:, :2].copy())
        if self.env.done:
            arrivals = [None if t is None else round(t, 2) for t in self.info["arrival_times"]]
            print(f"Seed {self.seed}: {self.status()} | arrivals={arrivals} | "
                  f"spread={self.info['arrival_spread']} | "
                  f"max deadline error={self.info['max_deadline_error']}", flush=True)

    def status(self):
        if self.info["collision"]:
            return "COLLISION"
        if self.info["success"]:
            return "SUCCESS — synchronized arrival"
        if self.info["all_arrived"]:
            return "ALL ARRIVED — timing missed"
        if self.info["timeout"]:
            return "TIMEOUT"
        return "PAUSED" if self.paused else "RUNNING"

    def draw(self):
        env, cfg = self.env, self.env.config
        history = np.asarray(self.history)
        body_local = np.array([[-0.10, -0.11], [cfg.wheelbase + 0.07, -0.11],
                               [cfg.wheelbase + 0.07, 0.11], [-0.10, 0.11]])
        for i, (x, y, yaw, speed, _) in enumerate(env.states):
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[c, -s], [s, c]])
            self.trails[i].set_data(history[:, i, 0], history[:, i, 1])
            self.bodies[i].set_xy(body_local @ rotation.T + [x, y])
            self.footprints[i].center = (x, y)
            self.headings[i].set_data([x, x + cfg.wheelbase * c], [y, y + cfg.wheelbase * s])
            self.labels[i].set_position((x + 0.15, y + 0.30))
            distance = np.linalg.norm(env.goals[i] - env.states[i, :2])
            arrival = self.info["arrival_times"][i]
            detail = f"arrived {arrival:.2f}s" if arrival is not None else f"goal {distance:.2f}m"
            self.car_info[i].set_text(f"Car {i + 1}   {speed:.2f} m/s   {detail}")
        needed = float(np.max(np.abs(env.states[:, :2]))) + 0.8
        if needed > self.extent:
            self.extent = needed + 0.5
            self.ax.set_xlim(-self.extent, self.extent)
            self.ax.set_ylim(-self.extent, self.extent)
        self.header.set_text(
            f"{self.status()}\nSeed {self.seed}  |  {env.family}\n"
            f"Time {env.time:.2f}s / deadline {env.deadline:.2f}s\n"
            f"Time remaining {env.deadline - env.time:+.2f}s\n"
            f"Playback {self.speed:g}×" + ("  |  PAUSED" if self.paused and env.done else ""))
        spread, error = self.info["arrival_spread"], self.info["max_deadline_error"]
        separation = self.info["min_separation"]
        summary = f"Parked: {int(env.arrived.sum())}/{env.n}\n"
        if separation is not None:
            summary += f"Minimum separation: {separation:.2f}m\n"
        if spread is not None:
            summary += f"Arrival spread: {spread:.2f}s (limit {cfg.synchrony_window:g})\n"
            summary += f"Deadline error: {error:.2f}s (limit {cfg.deadline_window:g})\n"
        if env.done:
            summary += ("N: new scenario · R: replay" if self.hold or self.paused
                        else f"Next scenario in {max(0, 3 - self.end_elapsed):.0f}s")
        self.summary.set_text(summary)
        self.figure.canvas.draw_idle()

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        self.last_wall = time.perf_counter()
        self.draw()

    def replay(self, event=None):
        self.reset(self.seed)

    def next_scenario(self, event=None):
        self.reset(self.seed + 1)

    def on_key(self, event):
        if event.key == " ":
            self.toggle_pause()
        elif event.key in ("n", "N"):
            self.next_scenario()
        elif event.key in ("r", "R"):
            self.replay()
        elif event.key in ("up", "down"):
            self.speed = float(np.clip(self.speed * (2 if event.key == "up" else 0.5), 0.125, 8))
            self.draw()
        elif event.key == "right":
            self.paused = True
            self.advance()
            self.draw()
        elif event.key in ("escape", "q"):
            self.plt.close(self.figure)

    def on_timer(self):
        now = time.perf_counter()
        wall_dt, self.last_wall = min(now - self.last_wall, 0.25), now
        if self.paused:
            return
        if self.env.done:
            if not self.hold:
                self.end_elapsed += wall_dt
                if self.end_elapsed >= 3:
                    self.next_scenario()
                else:
                    self.draw()
            return
        self.elapsed += wall_dt * self.speed
        for _ in range(4):
            if self.elapsed < self.env.config.policy_dt or self.env.done:
                break
            self.elapsed -= self.env.config.policy_dt
            self.advance()
        self.draw()

    def close(self, event=None):
        self.timer.stop()

    def show(self):
        self.timer.start()
        try:
            self.plt.show(block=True)
        finally:
            self.timer.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Default: newest runs/*/best.pt")
    parser.add_argument("--seed", type=int, help="Scenario seed; default: a fresh random seed")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback multiplier, 0.125 to 8")
    parser.add_argument("--hold", action="store_true", help="Keep the final frame instead of advancing")
    parser.add_argument("--cars", type=int, help="Override team size for a zero-shot test")
    parser.add_argument("--collision-model", choices=("discs", "single"),
                        help="discs = body-aligned circles (correct); single = legacy rear-axle circle")
    parser.add_argument("--footprint-discs", type=int, help="number of body discs when using discs")
    parser.add_argument("--neighbors", type=int, help="Override actor neighbor count")
    parser.add_argument("--controller", choices=("pp", "mpc"), help="Explicit controller override for comparison")
    parser.add_argument("--distribution", choices=("target", "training"), default="target",
                        help="Show final random distribution or the checkpoint's training stage")
    parser.add_argument("--family", choices=FAMILIES, help="Override the checkpoint scenario distribution")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not math.isfinite(args.speed) or not 0.125 <= args.speed <= 8:
        parser.error("--speed must be between 0.125 and 8")
    torch.set_num_threads(1)
    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        agent, config, metadata = load_policy(checkpoint, args.device, args.cars,
                                              args.neighbors, args.family, args.controller,
                                              args.distribution, args.collision_model,
                                              args.footprint_discs)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"} or "matplotlib_inline" in backend:
        parser.error("Matplotlib has no window backend. Run in a desktop terminal with "
                     "MPLBACKEND=MacOSX on macOS, or MPLBACKEND=TkAgg / QtAgg on Linux.")
    seed = args.seed if args.seed is not None else secrets.randbelow(2_000_000_000)
    print(f"Loaded {checkpoint}\nCheckpoint step: {metadata.get('global_step', 'unknown')}\n"
          "Space: pause | N: next | R: replay | right: step | up/down: speed | Esc: close", flush=True)
    viewer = LiveViewer(agent, config, checkpoint, metadata.get("global_step"), seed,
                        speed=args.speed, hold=args.hold)
    try:
        viewer.show()
    except KeyboardInterrupt:
        viewer.plt.close(viewer.figure)


if __name__ == "__main__":
    main()
