"""Six reproducible paired trajectory figures from the existing arrival_v2 policy.

Run from the repository root. No training or controller changes are performed.
  python3 arrival_final/paper/figures/trajectory_scenarios/generate.py --collect-only
  python3 arrival_final/paper/figures/trajectory_scenarios/generate.py --render-only

Rendering needs macOS graphics access. The saved NPZ files contain actual states
from independent kinematic and MuJoCo rollouts on identical sampled missions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/arrival_figure_mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

# Chosen before executing the policy; no screening for successful outcomes.
CASES = [
    ("01_random", "Random starts and goals", "random", 3, 7_710_001),
    ("02_crossing", "Three-way crossing", "crossing", 3, 7_710_002),
    ("03_converging", "Converging routes", "converging", 3, 7_710_003),
    ("04_parallel", "Parallel journeys", "parallel", 3, 7_710_004),
    ("05_crossing_five", "Five-way crossing", "crossing", 5, 7_710_005),
    ("06_crossing_eight", "Eight-way crossing", "crossing", 8, 7_710_006),
]
COLORS = ["#2274B5", "#E67E22", "#299D79", "#B74E73", "#8262AD",
          "#159FAD", "#B29A20", "#696F80"]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def collect(checkpoint):
    import torch
    from arrival_v2.config import EnvConfig
    from arrival_v2.env import ArrivalEnv
    from arrival_v2.mappo import MAPPOAgent
    from arrival_v2.plants import MujocoPlant

    torch.set_num_threads(1)
    agent, metadata = MAPPOAgent.from_checkpoint(checkpoint)
    agent.actor.eval()
    base = EnvConfig(**metadata["env_config"])
    manifest = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_training_step": metadata["global_step"],
        "runtime_package": "arrival_v2",
        "runtime_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "arrival_v2").glob("*.py"))
        },
        "checkpoint_env_config": metadata["env_config"],
        "runtime_default_fields_absent_from_checkpoint": {
            k: v for k, v in asdict(base).items() if k not in metadata["env_config"]
        },
        "selection": "Six fixed seeds chosen before rollout; no outcome screening.",
        "pairing": "Same initial poses, zero initial speeds, goals and deadline; independent closed-loop rollouts.",
        "sampling": "Recorded at each policy step (0.20 s), plus terminal state.",
        "purpose": "Illustrative episodes, not a statistical benchmark or safety certificate.",
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
        "scenarios": [],
    }
    for stem, title, family, n, seed in CASES:
        cfg = replace(base, n_cars=n, family=family, initial_speed_max=0.0).validate()
        case = {"stem": stem, "title": title, "seed": seed, "config": asdict(cfg), "runs": {}}
        first = None
        for kind in ("kinematic", "mujoco"):
            started = time.monotonic()
            plant = (MujocoPlant(cfg, scene_dir=str(OUT / "scenes" / stem))
                     if kind == "mujoco" else None)
            env = ArrivalEnv(cfg, plant=plant)
            observation = env.reset(seed=seed)
            if first is None:
                first = (env.states.copy(), env.goals.copy(), env.deadline)
                case["scenario_metadata"] = env.scenario.metadata
            else:
                np.testing.assert_array_equal(env.states, first[0])
                np.testing.assert_array_equal(env.goals, first[1])
                assert env.deadline == first[2]
            states, times, qpos = [], [], []
            done = False
            print(f"Simulating {stem}: {kind}, {n} cars, T={env.deadline:.2f}s", flush=True)
            while True:
                states.append(env.states.copy())
                times.append(env.time)
                if plant is not None:
                    qpos.append(plant.data.qpos.copy())
                if done:
                    break
                observation, _, done, info = env.step(agent.predict(observation))
            trace = dict(states=np.asarray(states), time=np.asarray(times),
                         goals=env.goals.copy(), deadline=np.array(env.deadline))
            if plant is not None:
                trace["qpos"] = np.asarray(qpos)
                trace["mocap_pos"] = plant.data.mocap_pos.copy()
                trace["mocap_quat"] = plant.data.mocap_quat.copy()
                plant.close()
            filename = f"{stem}_{kind}.npz"
            np.savez_compressed(OUT / "data" / filename, **trace)
            case["runs"][kind] = {"trace": "data/" + filename, "info": info}
            print(f"  {time.monotonic()-started:.1f}s wall time; reached {sum(info['arrived'])}/{n}; "
                  f"success={info['success']}; collision={info['collision']}", flush=True)
        manifest["scenarios"].append(case)
        save_json(OUT / "manifest.json", manifest)
    return manifest


def ghost_indices(states):
    """Select recorded earlier poses by distance travelled, without inventing poses."""
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))]
    if distance[-1] < 0.4:
        return []
    chosen = []
    for fraction in np.linspace(0., 0.88, 7):
        k = min(int(np.searchsorted(distance, fraction * distance[-1])), len(states)-2)
        if (np.linalg.norm(states[k, :2]-states[-1, :2]) > 0.45
                and (not chosen or np.linalg.norm(states[k, :2]-states[chosen[-1][0], :2]) > 0.45)):
            chosen.append((k, 0.13 + 0.45 * fraction))
    return chosen


def bounds_for(*traces):
    points = np.concatenate([np.r_[t["states"][:, :, :2].reshape(-1, 2), t["goals"]] for t in traces])
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo+hi)/2
    half = max(float((hi-lo).max()/2)+0.8, 2.3)
    return centre, half


def draw_car(ax, state, color, alpha=1.):
    from matplotlib.patches import Polygon
    x, y, yaw, _, steer = state
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

    def polygon(points, face, edge="none", zorder=5):
        ax.add_patch(Polygon(np.asarray(points) @ rotation.T + [x, y], closed=True,
                             facecolor=face, edgecolor=edge, linewidth=0.5,
                             alpha=alpha, zorder=zorder))

    # Actual XML chassis relative to rear axle, not an enlarged car icon.
    polygon([[-.105,-.09],[.26,-.09],[.43,-.055],[.43,.055],[.26,.09],[-.105,.09]], color,
            color if alpha < 1 else "#23415A")
    for wx in (0., .32226):
        angle = steer if wx > 0 else 0.
        turn = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        for wy in (-.122385,.122385):
            wheel = np.array([[-.0524,-.0222],[.0524,-.0222],[.0524,.0222],[-.0524,.0222]])
            polygon(wheel @ turn.T + [wx, wy], "#24303A", zorder=6)
    polygon([[.14,-.055],[.22,-.055],[.22,.055],[.14,.055]], "#F8FAFC", zorder=7)


def draw_2d(ax, trace, centre, half):
    from matplotlib.patches import Circle
    states, goals = trace["states"], trace["goals"]
    ax.set_xlim(centre[0]-half, centre[0]+half)
    ax.set_ylim(centre[1]-half, centre[1]+half)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFBFC")
    ax.grid(color="#E4E9EF", linewidth=.6, zorder=0)
    ax.set_xlabel("x [m]", color="#536273", fontsize=10)
    ax.set_ylabel("y [m]", color="#536273", fontsize=10)
    ax.tick_params(labelsize=9, colors="#718096", length=0)
    for spine in ax.spines.values():
        spine.set_color("#DDE4EA")
    for i in range(states.shape[1]):
        color = COLORS[i]
        path = states[:, i]
        ax.plot(path[:,0], path[:,1], color=color, linewidth=1.55, alpha=.85, zorder=2)
        for k, alpha in ghost_indices(path):
            draw_car(ax, path[k], color, alpha)
        ax.scatter(*path[0,:2], s=17, facecolors="white", edgecolors=color, linewidths=1., zorder=8)
        ax.annotate(f"S{i+1}", path[0,:2], xytext=(-8,8), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold", zorder=9)
        ax.add_patch(Circle(goals[i], .25, fill=False, edgecolor=color,
                            linewidth=1., linestyle=(0,(3,3)), zorder=3))
        ax.scatter(*goals[i], marker="+", s=48, color=color, linewidths=1.4, zorder=8)
        ax.annotate(f"G{i+1}", goals[i], xytext=(9,8), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold", zorder=9)
        draw_car(ax, path[-1], color)


def render_mujoco(case, trace, centre, half):
    import mujoco as mj
    from matplotlib.colors import to_rgb
    from arrival_v2.config import EnvConfig
    from arrival_v2.plants import MujocoPlant

    cfg = EnvConfig(**case["config"])
    plant = MujocoPlant(cfg, scene_dir=str(OUT / "scenes" / case["stem"]))
    model, data = plant.model, plant.data
    model.vis.global_.offwidth = 1600
    model.vis.global_.offheight = 1400
    floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = [.975, .980, .987, 1.]
    model.light_ambient[:] = [.08, .08, .08]
    model.light_diffuse[:] = [.45, .45, .45]
    model.light_specular[:] = [0., 0., 0.]
    model.vis.headlight.ambient[:] = [.25, .25, .25]
    model.vis.headlight.diffuse[:] = [.25, .25, .25]
    model.vis.headlight.specular[:] = [0., 0., 0.]
    car_geoms = []
    for i in range(cfg.n_cars):
        root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"chassis_{i}")
        geoms = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root]
        car_geoms.append(geoms)
        for g in geoms:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or ""
            if any(name.startswith(p) for p in ("chassis_", "front_", "back_", "bumper_")):
                model.geom_rgba[g] = [*to_rgb(COLORS[i]), 1.]
        target = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"target_{i}")
        model.geom_rgba[target,3] = 0.  # Draw exact tolerance rings instead of spheres.

    data.mocap_pos[:] = trace["mocap_pos"]
    data.mocap_quat[:] = trace["mocap_quat"]
    ghosts = []
    for i in range(cfg.n_cars):
        for k, alpha in ghost_indices(trace["states"][:,i]):
            data.qpos[:] = trace["qpos"][k]
            mj.mj_forward(model,data)
            for g in car_geoms[i]:
                ghosts.append((int(model.geom_type[g]), model.geom_size[g].copy(),
                               data.geom_xpos[g].copy(), data.geom_xmat[g].copy(),
                               np.r_[model.geom_rgba[g,:3],alpha]))
    data.qpos[:] = trace["qpos"][-1]
    mj.mj_forward(model,data)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [*centre, 0.]
    camera.azimuth = 90.
    camera.elevation = -72.
    camera.distance = 2.50 * half
    renderer = mj.Renderer(model,height=1400,width=1600,max_geom=12000)
    try:
        renderer.update_scene(data,camera)
        scene = renderer.scene
        scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = False
        scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False

        def geometry(kind, size, position, matrix, rgba, label=None):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("Rendering geometry capacity exceeded")
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(geom, kind, np.asarray(size,dtype=float),
                            np.asarray(position,dtype=float), np.asarray(matrix,dtype=float),
                            np.asarray(rgba,dtype=np.float32))
            geom.category = mj.mjtCatBit.mjCAT_DECOR
            if label:
                geom.label = label
            scene.ngeom += 1
            return geom

        def line(start,end,color,radius=.012):
            if np.linalg.norm(np.asarray(start)-end) < 1e-7:
                return
            geom = geometry(mj.mjtGeom.mjGEOM_CAPSULE,[radius,1.,1.],[0.,0.,0.],np.eye(3).ravel(),color)
            mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,radius,np.asarray(start),np.asarray(end))

        # Metric grid is decoration only; this environment has no walls.
        for x in np.arange(math.floor(centre[0]-half), math.ceil(centre[0]+half)+1):
            line([x,centre[1]-half,.003],[x,centre[1]+half,.003],[.86,.89,.92,1],.003)
        for y in np.arange(math.floor(centre[1]-half), math.ceil(centre[1]+half)+1):
            line([centre[0]-half,y,.003],[centre[0]+half,y,.003],[.86,.89,.92,1],.003)
        for i in range(cfg.n_cars):
            rgba = [*to_rgb(COLORS[i]), .95]
            path = trace["states"][:,i,:2]
            for a,b in zip(path[:-1],path[1:]):
                line([*a,.014],[*b,.014],rgba,.015)
            for xy, label, radius in ((path[0], f"S{i+1}", .07), (trace["goals"][i], f"G{i+1}", .25)):
                angles = np.linspace(0,2*np.pi,49)
                ring = xy + radius*np.column_stack([np.cos(angles),np.sin(angles)])
                for k,(a,b) in enumerate(zip(ring[:-1],ring[1:])):
                    if radius < .1 or k % 4 < 2:
                        line([*a,.018],[*b,.018],rgba,.011)
                geometry(mj.mjtGeom.mjGEOM_LABEL,[0,0,0],[xy[0]+.25,xy[1]+.2,.04],
                         np.eye(3).ravel(),rgba,label)
            gx,gy = trace["goals"][i]
            line([gx-.08,gy,.02],[gx+.08,gy,.02],rgba,.013)
            line([gx,gy-.08,.02],[gx,gy+.08,.02],rgba,.013)
        for kind,size,position,matrix,rgba in ghosts:
            geometry(kind,size,position,matrix,rgba)
        return renderer.render().copy()
    finally:
        renderer.close()
        plant.close()


def result_label(info):
    reached = sum(info["arrived"])
    total = len(info["arrived"])
    status = ("SYNCHRONIZED" if info["success"] else "COLLISION" if info["collision"] else
              "TIMEOUT" if info["timeout"] else "TIMING MISSED")
    spread = info["arrival_spread"]
    timing = f"arrival spread {spread:.2f} s" if spread is not None else "arrival spread unavailable"
    return f"{status}  |  reached {reached}/{total}  |  {timing}"


def render(manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image, ImageOps, ImageDraw

    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none"})
    images = []
    for case in manifest["scenarios"]:
        traces = {}
        for kind, run in case["runs"].items():
            with np.load(OUT / run["trace"],allow_pickle=False) as a:
                traces[kind] = {k:a[k] for k in a.files}
        kin, mjc = traces["kinematic"],traces["mujoco"]
        centre,half = bounds_for(kin,mjc)
        native = render_mujoco(case,mjc,centre,half)
        Image.fromarray(native).save(OUT / "panels" / f"{case['stem']}_mujoco.png")
        fig = plt.figure(figsize=(14.4,8.3),facecolor="white")
        ax = fig.add_axes([.055,.22,.415,.63])
        right = fig.add_axes([.515,.22,.455,.63])
        draw_2d(ax,kin,centre,half)
        right.imshow(native)
        right.axis("off")
        fig.text(.055,.955,case["stem"][:2],color="#8C9AA8",fontsize=19,weight="bold")
        fig.text(.095,.955,case["title"],color="#21384D",fontsize=22,weight="bold")
        fig.text(.055,.915,f"{case['config']['n_cars']} vehicles   /   shared deadline T = {float(kin['deadline']):.2f} s"
                 "   /   identical starting scenario",fontsize=11,color="#687887")
        fig.text(.055,.875,"(a)  Kinematic bicycle",fontsize=12,weight="bold",color="#263C50")
        fig.text(.545,.875,"(b)  MuJoCo contact dynamics",fontsize=12,weight="bold",color="#263C50")
        fig.text(.055,.147,result_label(case['runs']['kinematic']['info']),fontsize=9,color="#405568")
        fig.text(.545,.147,result_label(case['runs']['mujoco']['info']),fontsize=9,color="#405568")
        handles = [Line2D([0],[0],color=COLORS[i],lw=3,label=f"Car {i+1}") for i in range(case['config']['n_cars'])]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.071),ncol=len(handles),
                   frameon=False,fontsize=9,handlelength=1.6,columnspacing=1.5)
        fig.text(.5,.054,"Lines: recorded paths     Faded cars: earlier poses     Solid cars: final poses     S: start     G: assigned goal",
                 ha="center",fontsize=9,color="#617080")
        fig.text(.5,.022,"Frozen MAPPO + pure-pursuit references + NMPC  |  Independent closed-loop rollouts  |  Illustrative episode",
                 ha="center",fontsize=8,color="#8B98A5")
        destination = OUT / f"{case['stem']}.png"
        fig.savefig(destination,dpi=220,facecolor="white")
        fig.savefig(OUT / f"{case['stem']}.svg",facecolor="white")
        plt.close(fig)
        images.append(destination)
        print(f"Rendered {destination.name}",flush=True)
    sheet = Image.new("RGB",(1800,1620),"#EDF1F5")
    for i,path in enumerate(images):
        with Image.open(path) as im:
            thumb=ImageOps.contain(im.convert("RGB"),(885,510))
            sheet.paste(thumb,((i%2)*900+7,(i//2)*540+12))
    sheet.save(OUT / "overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=ROOT / "runs/mpc_s0/best.pt")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--collect-only",action="store_true")
    group.add_argument("--render-only",action="store_true")
    args=parser.parse_args()
    for folder in ("data","panels","scenes"):
        (OUT / folder).mkdir(parents=True,exist_ok=True)
    manifest=(json.loads((OUT / "manifest.json").read_text()) if args.render_only
              else collect(args.checkpoint.resolve()))
    if not args.collect_only:
        render(manifest)


if __name__ == "__main__":
    main()
