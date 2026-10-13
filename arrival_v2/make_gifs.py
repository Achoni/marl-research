"""Render demo GIFs of a trained policy, for slides and for the supervisor.

    python -m arrival_v2.make_gifs --checkpoint runs/mpc_s0/best.pt

Writes into arrival_v2/videos/ :

    mujoco/       MuJoCo F1TENTH cars, rendered from the simulator's own camera
    bicycle/      the kinematic training model, top-down
    pure_pursuit/ the same policy with the MPC REMOVED, top-down

one GIF per team size (3, 5, 8 cars), each containing 10 consecutive episodes with
a caption naming the scenario and the outcome. The three folders use the SAME
episode seeds, so the same row of any two folders is the same scenario.

Useful flags:
    --episodes 10      episodes per GIF
    --cars 3 5 8       team sizes
    --only mujoco      render just one folder
    --fps 12           playback speed
"""
import argparse
import math
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import ArrivalEnv
from .mappo import MAPPOAgent
from .model import body_discs

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
SEED_BASE = 7_700_000


def outcome(info):
    if info["success"]:
        return "SUCCESS"
    if info["collision"]:
        return "COLLISION"
    if info["timeout"]:
        return "TIMEOUT"
    return "off-deadline"


def rollout(env, agent, seed, mujoco_renderer=None, on_reset=None):
    """Play one episode, keeping a frame per policy step."""
    observation = env.reset(seed=seed)
    if on_reset is not None:
        on_reset(env)
    frames, states = [], []
    done = False
    while True:
        states.append((env.states.copy(), env.arrived.copy(), env.time))
        if mujoco_renderer is not None:
            frames.append(mujoco_renderer(env))
        if done:
            break
        observation, reward, done, info = env.step(agent.predict(observation))
    return states, frames, info


# ---------------------------------------------------------------- top-down (matplotlib)
def draw_topdown(axis, plt, cfg, snapshot, history, goals, info, title):
    axis.clear()
    states, arrived, now = snapshot
    offsets, radius = body_discs(cfg)
    extent = max(np.abs(np.r_[states[:, :2], goals]).max() + 1.0, 3.0)
    axis.set_xlim(-extent, extent); axis.set_ylim(-extent, extent)
    axis.set_aspect("equal"); axis.set_xticks([]); axis.set_yticks([]); axis.grid(alpha=0.12)
    axis.set_title(title, fontsize=11)
    for i in range(len(states)):
        color = COLORS[i % len(COLORS)]
        axis.plot(*goals[i], marker="x", ms=10, mew=2.2, color=color)
        axis.add_patch(plt.Circle(goals[i], cfg.goal_tolerance, fill=False, ls=":",
                                  color=color, alpha=0.65))
        if len(history) > 1:
            trail = np.array([h[i, :2] for h in history])
            axis.plot(trail[:, 0], trail[:, 1], color=color, lw=1.1, alpha=0.4)
        heading = np.array([math.cos(states[i, 2]), math.sin(states[i, 2])])
        for offset in offsets:                      # the actual collision footprint
            axis.add_patch(plt.Circle(states[i, :2] + offset * heading, radius, color=color,
                                      alpha=0.75 if not arrived[i] else 0.3, lw=0))
        nose = states[i, :2] + (offsets.max() + radius) * heading
        axis.plot([states[i, 0], nose[0]], [states[i, 1], nose[1]], "k-", lw=1.4)
    return []


def render_topdown(path, episodes, cfg, fps, subtitle, stride=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    timeline = [(e, k) for e in episodes
                for k in list(range(0, len(e["states"]), stride)) + [len(e["states"]) - 1] * max(fps // 2, 1)]
    figure, axis = plt.subplots(figsize=(6.2, 6.2))

    def draw(step):
        episode, index = timeline[step]
        index = min(index, len(episode["states"]) - 1)
        finished = index >= len(episode["states"]) - 1
        snapshot = episode["states"][index]
        title = (f"episode {episode['number']}/{episode['total']}   {subtitle}\n"
                 f"{episode['family']}   t={snapshot[2]:4.1f}s / T={episode['deadline']:.1f}s"
                 + (f"   {outcome(episode['info'])}" if finished else ""))
        return draw_topdown(axis, plt, cfg, snapshot,
                            [s[0] for s in episode["states"][:index + 1]],
                            episode["goals"], episode["info"], title)

    FuncAnimation(figure, draw, frames=len(timeline), interval=1000 / fps, blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(figure)


# ---------------------------------------------------------------- MuJoCo (offscreen)
def render_mujoco(path, episodes, fps, stride=1):
    import imageio.v2 as imageio
    frames = []
    for episode in episodes:
        frames.extend(episode["frames"][::stride])
        frames.extend([episode["frames"][-1]] * max(fps // 2, 1))   # hold on the last frame
    imageio.mimsave(path, frames, fps=fps, loop=0)


def build(kind, checkpoint, cars, episodes, fps, out_root, footprint_discs,
          stride=1, width=560, height=420, mujoco_3d=False):
    from dataclasses import replace
    loaded = MAPPOAgent.from_checkpoint(checkpoint)
    agent, metadata = loaded if isinstance(loaded, tuple) else (loaded, {})
    agent.actor.eval()
    base = EnvConfig(**metadata["env_config"])
    base = replace(base, collision_model="discs", footprint_discs=footprint_discs)

    folder = out_root / kind
    folder.mkdir(parents=True, exist_ok=True)
    subtitle = dict(mujoco="MuJoCo plant, NMPC",
                    bicycle="kinematic plant, NMPC",
                    pure_pursuit="kinematic plant, NO MPC")[kind]

    for n in cars:
        cfg = replace(base, n_cars=n,
                      controller="pp" if kind == "pure_pursuit" else "mpc").validate()
        plant = renderer = camera = None
        if kind == "mujoco":
            pass
        if kind == "mujoco":
            from .plants import MujocoPlant
            import mujoco as mj
            plant = MujocoPlant(cfg)
            renderer = mj.Renderer(plant.model, height=height, width=width) if mujoco_3d else None
            # A checkerboard floor is high-frequency detail that GIF cannot compress:
            # it made a 2-episode clip 11 MB. A flat floor cuts that by an order of
            # magnitude and reads better from above anyway.
            floor = mj.mj_name2id(plant.model, mj.mjtObj.mjOBJ_GEOM, "floor")
            if floor >= 0:
                plant.model.geom_matid[floor] = -1
                plant.model.geom_rgba[floor] = [0.92, 0.93, 0.95, 1.0]
            plant.model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
            # Recolour cars and their targets from one bright palette so each car is
            # unmistakably paired with its own goal, and enlarge the targets: the stock
            # 0.10 m spheres are invisible from above.
            palette = [(0.11, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
                       (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
                       (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
                       (0.09, 0.75, 0.81)]
            for i in range(n):
                red, green, blue = palette[i % len(palette)]
                for part in ("chassis_b", "chassis_f", "front", "back", "bumper"):
                    geom = mj.mj_name2id(plant.model, mj.mjtObj.mjOBJ_GEOM, f"{part}_{i}")
                    if geom >= 0:
                        plant.model.geom_rgba[geom] = [red, green, blue, 1.0]
                target = mj.mj_name2id(plant.model, mj.mjtObj.mjOBJ_GEOM, f"target_{i}")
                if target >= 0:
                    plant.model.geom_size[target] = [cfg.goal_tolerance, 0.0, 0.0]
                    plant.model.geom_rgba[target] = [red, green, blue, 0.45]
            camera = mj.MjvCamera()
            camera.type = mj.mjtCamera.mjCAMERA_FREE
            camera.azimuth = 90.0
            camera.elevation = -80.0        # near top-down, so the geometry reads clearly

        env = ArrivalEnv(cfg, plant=plant)

        smoothed = {}

        def frame_camera(env):
            """Track the cars and their goals so they stay large in frame.

            Fitting the whole episode once makes a 0.54 m car ~20 px tall. Following
            the live bounding box of cars-plus-unreached-goals, smoothed so it does not
            jitter, keeps them 2-3x larger.
            """
            pending = env.goals[~env.arrived] if (~env.arrived).any() else env.goals
            points = np.r_[env.states[:, :2], pending]
            centre = 0.5 * (points.max(axis=0) + points.min(axis=0))
            extent = float(np.abs(points - centre).max()) + 1.0
            # fovy is 45 deg, so the visible half-height is d*tan(22.5) = 0.414*d.
            # To keep a point `extent` from the centre inside the frame we need
            # d >= extent/0.414 = 2.41*extent; 2.6 leaves a little margin.
            distance = 2.6 * max(extent, 1.2)
            if smoothed:                       # exponential smoothing, no jitter
                centre = 0.85 * smoothed["centre"] + 0.15 * centre
                distance = 0.85 * smoothed["distance"] + 0.15 * distance
            smoothed["centre"], smoothed["distance"] = centre, distance
            camera.lookat[:] = [centre[0], centre[1], 0.0]
            camera.distance = distance

        def snap(env=None):
            if env is not None:
                frame_camera(env)
            renderer.update_scene(plant.data, camera)
            return renderer.render()

        collected, summary = [], []
        for index in range(episodes):
            want_3d = kind == "mujoco" and mujoco_3d
            states, frames, info = rollout(env, agent, SEED_BASE + index,
                                           snap if want_3d else None,
                                           frame_camera if want_3d else None)
            collected.append(dict(number=index + 1, total=episodes, states=states, frames=frames,
                                  goals=env.goals.copy(), deadline=info["deadline"],
                                  family=info["family"], info=info))
            summary.append(info)
            spread = info["arrival_spread"]
            spread_text = "n/a" if spread is None else f"{spread:.2f}s"
            print(f"    {kind:>12} N={n} episode {index + 1:2d}/{episodes}  "
                  f"{outcome(info):13s} spread={spread_text}", flush=True)

        path = folder / f"{n}_cars.gif"
        if kind == "mujoco" and mujoco_3d:
            render_mujoco(path, collected, fps, stride)
        else:
            # Top-down even for MuJoCo: the states ARE MuJoCo's, and a 0.5 m car in a
            # 13 m arena is unreadable in perspective. This also makes the three
            # folders directly comparable frame for frame.
            render_topdown(path, collected, cfg, fps, subtitle, stride)
        if renderer is not None:
            renderer.close()
        if plant is not None:
            plant.close()
        rate = np.mean([s["success"] for s in summary])
        crash = np.mean([s["collision"] for s in summary])
        spreads = [s["arrival_spread"] for s in summary if s["arrival_spread"] is not None]
        print(f"  -> {path}   success {rate:.0%}  collisions {crash:.0%}  "
              f"spread {np.mean(spreads) if spreads else float('nan'):.2f}s\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/mpc_s0/best.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--cars", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--fps", type=int, default=10)   # 1.2x slower than the old 12
    parser.add_argument("--footprint-discs", type=int, default=3)
    parser.add_argument("--only", choices=("mujoco", "bicycle", "pure_pursuit"))
    parser.add_argument("--mujoco-topdown", action="store_true",
                        help="draw the MuJoCo folder top-down instead of from the simulator camera")
    parser.add_argument("--stride", type=int, default=3,
                        help="keep every Nth policy step; 2 halves the file size")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "videos")
    arguments = parser.parse_args()

    kinds = [arguments.only] if arguments.only else ["bicycle", "pure_pursuit", "mujoco"]
    print(f"checkpoint {arguments.checkpoint}\noutput     {arguments.out}\n"
          f"{arguments.episodes} episodes per GIF, team sizes {arguments.cars}, "
          f"{arguments.footprint_discs} body discs\n")
    for kind in kinds:
        build(kind, arguments.checkpoint, arguments.cars, arguments.episodes,
              arguments.fps, arguments.out, arguments.footprint_discs,
              arguments.stride, arguments.width, arguments.height, not arguments.mujoco_topdown)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
