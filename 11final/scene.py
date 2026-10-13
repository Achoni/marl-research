"""Programmatic MuJoCo scene generator -- the scalability infrastructure.

Stamps out N cars / N targets / M obstacles into a scene the env loads, so the
same code runs 3 cars or 12 cars with no hand-edited XML. The per-car block is
templated from the hand-built car0.xml (index 0) by substituting the car index.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# one car, index 0 -- names carry the "_0" / "rf0_" / "ld0_" index tokens that
# get rewritten per car. Lidar sites are kept (harmless) but the rangefinder
# sensor stays off; obstacles are sensed via relative features in the obs.
CAR_TEMPLATE = """<mujoco>
  <worldbody>
    <body name="chassis_0" pos="{px} {py} 0.052388" quat="0.609 0 0 -0.793">
      <inertial pos="0 0 0" diaginertia=".05 .05 .08" mass="3.0"/>
      <joint name="car_0_free" type="free"/>
      <geom name="chassis_b_0" type="box" size=".10113 .1016 .02" pos="-.06 0 0"      rgba="{r} {g} {b} 1"/>
      <geom name="chassis_f_0" type="box" size=".06 .07 .02"      pos=".10113 0 0"     rgba="{r} {g} {b} 1"/>
      <geom name="front_0"     type="box" size=".052388 .02 .02"  pos=".2135 0 0"      rgba="{r} {g} {b} 1"/>
      <geom name="back_0"      type="box" size=".052388 .02 .02"  pos="-.2135 0 0"     rgba="{r} {g} {b} 1"/>
      <geom name="bumper_0"    type="box" size=".005 .09 .02"     pos=".265888 0 0.02" rgba="{r} {g} {b} 1"/>
      <body name="lidar_0" pos=".12 0 .08" euler="0 0 -135">
        <geom name="lidar_geom_0" type="cylinder" size=".01984 .02" mass="0" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="car_0_wheel_fl">
        <joint name="car_0_wfl_steer" type="hinge" pos="0.16113 .10016 0" limited="true" frictionloss="0.2" damping="0.2" armature="0.001" range="-30 30" axis="0 0 1"/>
        <joint name="car_0_wfl" type="hinge" pos="0.16113 .122385 0" axis="0 1 0" frictionloss="0.1" damping="0.05" armature="0.01" limited="false"/>
        <geom name="car_0_wfl" type="cylinder" size=".052388 .022225" pos="0.16113 .122385 0" mass="0.1" zaxis="0 1 0"/>
      </body>
      <body name="car_0_wheel_rl">
        <joint name="car_0_wrl" type="hinge" pos="-0.16113 .122385 0" axis="0 1 0" frictionloss="0.1" damping="0.05" armature="0.01" limited="false"/>
        <geom name="car_0_wrl" type="cylinder" size=".052388 .022225" pos="-0.16113 .122385 0" mass="0.1" zaxis="0 1 0"/>
      </body>
      <body name="car_0_wheel_fr">
        <joint name="car_0_wfr_steer" type="hinge" pos="0.16113 -.10016 0" limited="true" frictionloss="0.2" damping="0.2" armature="0.001" range="-30 30" axis="0 0 1"/>
        <joint name="car_0_wfr" type="hinge" pos="0.16113 -.122385 0" axis="0 1 0" frictionloss="0.1" damping="0.05" armature="0.01" limited="false"/>
        <geom name="car_0_wfr" type="cylinder" size=".052388 .022225" pos="0.16113 -.122385 0" mass="0.1" zaxis="0 1 0"/>
      </body>
      <body name="car_0_wheel_rr">
        <joint name="car_0_wrr" type="hinge" pos="-0.16113 -.122385 0" axis="0 1 0" frictionloss="0.1" damping="0.05" armature="0.01" limited="false"/>
        <geom name="car_0_wrr" type="cylinder" size=".052388 .022225" pos="-0.16113 -.122385 0" mass="0.1" zaxis="0 1 0"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="car_0_wfl_steer" joint2="car_0_wfr_steer" polycoef="0 1 0 0 0"/>
  </equality>
  <tendon>
    <fixed name="drive_0">
      <joint joint="car_0_wfl" coef="1"/>
      <joint joint="car_0_wrl" coef="1"/>
      <joint joint="car_0_wfr" coef="1"/>
      <joint joint="car_0_wrr" coef="1"/>
    </fixed>
  </tendon>
  <contact>
    <pair geom1="car_0_wfl" geom2="floor" condim="6" friction="2.5 2.5 .009 .0001 .0001"/>
    <pair geom1="car_0_wfr" geom2="floor" condim="6" friction="2.5 2.5 .009 .0001 .0001"/>
    <pair geom1="car_0_wrl" geom2="floor" condim="6" friction="2.5 2.5 .009 .0001 .0001"/>
    <pair geom1="car_0_wrr" geom2="floor" condim="6" friction="2.5 2.5 .009 .0001 .0001"/>
  </contact>
  <actuator>
    <motor name="drive_0" tendon="drive_0"/>
    <position forcelimited="true" forcerange="-5 5" name="steer_0" joint="car_0_wfl_steer" kp="15"/>
  </actuator>
</mujoco>
"""

# distinct colours cycled across cars (purely cosmetic)
_PALETTE = [(0.20, 0.20, 0.25), (0.80, 0.20, 0.20), (0.20, 0.35, 0.80),
            (0.20, 0.65, 0.30), (0.85, 0.55, 0.10), (0.55, 0.25, 0.70)]


def _car_xml(i, px, py):
    r, g, b = _PALETTE[i % len(_PALETTE)]
    block = CAR_TEMPLATE.format(px=px, py=py, r=r, g=g, b=b)
    # rewrite the index tokens (_0 only appears in names, never in numeric literals)
    return block.replace("_0", f"_{i}").replace("rf0_", f"rf{i}_").replace("ld0_", f"ld{i}_")


def _map_xml(n_targets, n_obst):
    tcol = [(0.20, 0.20, 0.25), (0.80, 0.20, 0.20), (0.20, 0.35, 0.80),
            (0.20, 0.65, 0.30), (0.85, 0.55, 0.10), (0.55, 0.25, 0.70)]
    targets = "\n".join(
        f'        <body name="target_{i}" mocap="true" pos="{1.5 - 0.3 * i} 0 0.05">'
        f'<geom name="target_{i}" type="sphere" size="0.10" '
        f'rgba="{tcol[i % len(tcol)][0]} {tcol[i % len(tcol)][1]} {tcol[i % len(tcol)][2]} 1" '
        f'contype="0" conaffinity="0"/></body>'
        for i in range(n_targets))
    obst = "\n".join(
        f'        <body name="obst_{k}" mocap="true" pos="{4 - k} 4 0.4">'
        f'<geom name="obst_{k}" type="cylinder" size="0.35 0.4" '
        f'rgba="0.55 0.35 0.15 1" contype="1" conaffinity="1"/></body>'
        for k in range(n_obst))
    return f"""<mujoco model="map">
    <visual><global offwidth="1280" offheight="960"/></visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                 rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>
    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="50 50 0.1" material="groundplane"/>
        <geom name="wall_north" type="box" size="12.1 0.1 0.3" pos="0 12 0"  rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_south" type="box" size="12.1 0.1 0.3" pos="0 -12 0" rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_east"  type="box" size="0.1 12 0.3"  pos="12 0 0"  rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_west"  type="box" size="0.1 12 0.3"  pos="-12 0 0" rgba="0.5 0.5 0.5 1"/>
{targets}
{obst}
    </worldbody>
</mujoco>
"""


def generate(n_cars, n_targets=None, n_obst=0, out_dir=None):
    """Write map/car/scene XML for N cars and return the scene path."""
    n_targets = n_cars if n_targets is None else n_targets
    out_dir = out_dir or os.path.join(HERE, "generated")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "map.xml"), "w") as f:
        f.write(_map_xml(n_targets, n_obst))
    includes = ['<include file="map.xml"/>']
    for i in range(n_cars):
        px, py = 3.0 + 0.8 * i, 0.3          # distinct load positions (reset overwrites)
        with open(os.path.join(out_dir, f"car{i}.xml"), "w") as f:
            f.write(_car_xml(i, px, py))
        includes.append(f'<include file="car{i}.xml"/>')
    scene = "<mujoco model=\"scene\">\n    " + "\n    ".join(includes) + "\n</mujoco>\n"
    scene_path = os.path.join(out_dir, "scene.xml")
    with open(scene_path, "w") as f:
        f.write(scene)
    return scene_path


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print("wrote", generate(n, n_obst=m))
