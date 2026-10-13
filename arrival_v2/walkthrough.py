"""One policy step, traced end to end with real numbers from the real code.

    python -m arrival_v2.walkthrough                 # the default worked example
    python -m arrival_v2.walkthrough --no-neighbour  # same, with nothing to avoid

Prints, in order:
    1. the state the car is in and what the policy sees
    2. the three numbers the policy outputs, and how they decode
    3. how pure pursuit turns them into a 1.2 s timed reference (the arithmetic)
    4. how the NMPC picks a control for the first 0.05 s tick, and why it differs
    5. the physics: five 0.01 s sub-steps to the next state
    6. ticks 2-4, then the state the policy sees 0.2 s later

Nothing here is a simplification of the pipeline; it calls the same functions the
environment calls.
"""
import argparse
import math

import numpy as np

from .config import EnvConfig
from .model import bicycle_step, body_discs, disc_centres
from .mpc import MPCTracker
from .reference import ReferenceGenerator, decode_action, pure_pursuit


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-neighbour", action="store_true")
    args = parser.parse_args()

    cfg = EnvConfig(controller="mpc").validate()
    generator = ReferenceGenerator(cfg)
    tracker = MPCTracker(cfg)

    # ---- the situation ------------------------------------------------------------------
    state = np.array([0.00, 0.00, 0.00, 1.00, 0.000])     # x, y, yaw, speed, steering
    goal = np.array([5.0, 2.0])
    deadline, elapsed = 9.0, 2.0
    neighbour = np.array([[2.10, 0.55, math.radians(200), 0.80, 0.0]])
    if args.no_neighbour:
        neighbour = None

    rule("1.  WHERE THE CAR IS, AND WHAT THE POLICY SEES")
    print(f"  state x = [x={state[0]:.2f} m, y={state[1]:.2f} m, yaw={math.degrees(state[2]):.0f}deg, "
          f"v={state[3]:.2f} m/s, steer={math.degrees(state[4]):.0f}deg]")
    print(f"  its goal            : {goal}  ->  {np.linalg.norm(goal - state[:2]):.2f} m away")
    print(f"  shared deadline T   : {deadline:.1f} s, of which {elapsed:.1f} s has elapsed "
          f"-> {deadline - elapsed:.1f} s left")
    print(f"  required avg speed  : {np.linalg.norm(goal - state[:2]) / (deadline - elapsed):.3f} m/s"
          "   <- observation feature 8")
    if neighbour is not None:
        rel = neighbour[0, :2] - state[:2]
        print(f"  one neighbour at    : {neighbour[0, :2]}, {np.linalg.norm(rel):.2f} m away, "
              f"heading {math.degrees(neighbour[0, 2]):.0f}deg at {neighbour[0, 3]:.2f} m/s "
              f"(coming toward us)")

    # ---- what the policy says -----------------------------------------------------------
    action = np.array([0.20, 0.35, 0.10])
    waypoint, preferred = decode_action(state, action, cfg)
    rule("2.  WHAT THE POLICY OUTPUTS  (three numbers, every 0.2 s)")
    forward = cfg.waypoint_forward_min + 0.5 * (action[0] + 1) * (cfg.waypoint_forward_max - cfg.waypoint_forward_min)
    print(f"  raw action           : [{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}]   all in [-1, 1]")
    print(f"  a0 -> forward offset : 0.25 + (({action[0]:+.2f})+1)/2 * (2.00-0.25) = {forward:.3f} m ahead")
    print(f"  a1 -> lateral offset : {action[1]:+.2f} * 1.20                        = {action[1] * cfg.waypoint_lateral_max:+.3f} m to the left")
    print(f"  a2 -> preferred speed: (({action[2]:+.2f})+1)/2 * 2.00                = {preferred:.3f} m/s")
    print(f"\n  those offsets are in the CAR's frame; rotated into the world (yaw={math.degrees(state[2]):.0f}deg):")
    print(f"  waypoint = {np.round(waypoint, 3)}")
    print("\n  NOTE: this is a via point to steer THROUGH, not a place to stop, and it is")
    print("        thrown away and replaced 0.2 s from now.")

    # ---- pure pursuit builds the timed reference -----------------------------------------
    states, controls = generator(state, action, goal)
    rule("3.  PURE PURSUIT TURNS THAT INTO A TIMED REFERENCE  (24 steps of 0.05 s)")
    print("  It simulates the bicycle model forward from the MEASURED state. At each step:")
    print("      steer_target = atan( 2 * L * lateral_offset_to_target / distance^2 )")
    print("      accel        = 4.0 * (target_speed - current_speed)")
    print("      target_speed = min(preferred, sqrt(2*b*dist_to_goal), sqrt(2*b*dist_to_waypoint))")
    print("\n  the arithmetic for step k=0, worked through:")
    dx, dy = waypoint - state[:2]
    lateral = -math.sin(state[2]) * dx + math.cos(state[2]) * dy
    d2 = dx * dx + dy * dy
    steer_target = math.atan2(2.0 * cfg.wheelbase * lateral, d2)
    d_goal = np.linalg.norm(goal - state[:2])
    d_way = np.linalg.norm(waypoint - state[:2])
    cap_goal = math.sqrt(2 * cfg.max_brake * max(0.0, d_goal - 0.5 * cfg.goal_tolerance))
    cap_way = math.sqrt(2 * cfg.max_brake * d_way)
    target_speed = min(preferred, cap_goal, cap_way)
    print(f"      target is {d_way:.3f} m away, {lateral:+.3f} m to the left")
    print(f"      steer_target = atan(2*{cfg.wheelbase}*{lateral:+.3f} / {d2:.3f}) = {math.degrees(steer_target):+.2f}deg")
    print(f"      speed caps   : preferred {preferred:.3f} | goal-brake {cap_goal:.3f} | waypoint-brake {cap_way:.3f}"
          f"  -> target {target_speed:.3f} m/s")
    print(f"      accel        = 4.0 * ({target_speed:.3f} - {state[3]:.3f}) = {4.0*(target_speed-state[3]):+.3f}"
          f"  -> clipped to {controls[0][0]:+.3f} m/s^2")
    print(f"      steer rate   = ({math.degrees(steer_target):+.2f} - {math.degrees(state[4]):+.2f})/0.05"
          f" = {math.degrees((steer_target-state[4])/cfg.control_dt):+.1f} deg/s"
          f"  -> clipped to {math.degrees(controls[0][1]):+.1f} deg/s   <-- SATURATES the 90 deg/s limit")
    print("\n  the resulting reference. row k says WHERE TO BE at t0 + k*0.05 s:")
    print(f"     {'k':>3} {'t (s)':>7} {'x':>8} {'y':>8} {'yaw':>8} {'v':>7} {'steer':>8} | {'accel':>7} {'rate':>9}")
    for k in list(range(0, 5)) + [8, 12, 16, 20, 24]:
        s = states[k]
        u = controls[min(k, len(controls) - 1)]
        print(f"     {k:3d} {k*cfg.control_dt:7.2f} {s[0]:8.3f} {s[1]:8.3f} {math.degrees(s[2]):7.1f}d "
              f"{s[3]:7.3f} {math.degrees(s[4]):7.1f}d | {u[0]:+7.3f} {math.degrees(u[1]):+8.1f}d")

    # ---- the MPC ------------------------------------------------------------------------
    rule("4.  THE NMPC PICKS THE CONTROL FOR THE FIRST 0.05 s TICK")
    offsets, radius = body_discs(cfg)
    print(f"  It optimises 32 numbers (accel, steer rate for 16 steps of 0.05 s) to minimise")
    print(f"      8|pos err|^2 + 3(speed err)^2 + 0.5(yaw err)^2 + 0.1(steer err)^2")
    print(f"      + 0.015|u|^2 + 0.04|u - u_prev|^2,  with x10/x2 weights on the final step")
    print(f"  subject to staying {2*radius + cfg.collision_margin:.3f} m from every sensed car's discs.")
    print(f"\n  CRUCIAL: the cost reads only the reference STATES above. The reference CONTROLS")
    print(f"  are used only as the solver's starting guess. So the MPC is free to disagree.")
    tracker.reset()
    command = tracker.control(state, states, controls, offset=0, neighbors=neighbour)
    print(f"\n  reference control for this tick : accel {controls[0][0]:+.3f}   rate {math.degrees(controls[0][1]):+7.1f} deg/s")
    print(f"  what the MPC actually commands  : accel {command[0]:+.3f}   rate {math.degrees(command[1]):+7.1f} deg/s")
    print(f"  solve time {1000*tracker.last_info['solve_time']:.1f} ms, "
          f"{tracker.last_info['iterations']} iterations, "
          f"fallback={tracker.last_info['fallback']}, neighbours seen={tracker.last_info['neighbor_count']}")
    print(f"\n  WHY THEY DIFFER: pure pursuit demands the whole steering change in ONE step and")
    print(f"  saturates the rate limit. The MPC's effort and smoothness terms make that")
    print(f"  expensive, and it can track nearly as well by turning more gently.")
    if neighbour is not None:
        print(f"  It is ALSO holding the separation constraint against the oncoming car.")

    # ---- the plant ----------------------------------------------------------------------
    rule("5.  THE PLANT: FIVE 0.01 s SUB-STEPS WITH THAT ONE COMMAND HELD")
    print(f"     {'sub':>4} {'t (s)':>7} {'x':>8} {'y':>8} {'yaw':>8} {'v':>7} {'steer':>8}")
    current = state.copy()
    print(f"     {'in':>4} {0.00:7.2f} {current[0]:8.4f} {current[1]:8.4f} "
          f"{math.degrees(current[2]):7.2f}d {current[3]:7.4f} {math.degrees(current[4]):7.2f}d")
    for sub in range(5):
        current = bicycle_step(current, command, cfg.physics_dt, cfg)
        print(f"     {sub+1:4d} {(sub+1)*cfg.physics_dt:7.2f} {current[0]:8.4f} {current[1]:8.4f} "
              f"{math.degrees(current[2]):7.2f}d {current[3]:7.4f} {math.degrees(current[4]):7.2f}d")
    print(f"\n  reference said be at   x={states[1][0]:.4f} y={states[1][1]:.4f} v={states[1][3]:.4f}")
    print(f"  the car actually is at x={current[0]:.4f} y={current[1]:.4f} v={current[3]:.4f}")
    print(f"  tracking error after one tick: {np.linalg.norm(current[:2] - states[1][:2])*100:.2f} cm, "
          f"{abs(current[3]-states[1][3]):.4f} m/s")

    # ---- the remaining ticks ------------------------------------------------------------
    rule("6.  TICKS 2-4: SAME REFERENCE, RE-SOLVED FROM WHERE THE CAR NOW IS")
    print(f"     {'tick':>5} {'starts from x,y':>22} {'MPC accel':>11} {'MPC rate':>11} {'ref row':>9} {'err (cm)':>10}")
    print(f"     {1:5d} {f'({state[0]:.3f}, {state[1]:.3f})':>22} {command[0]:+11.3f} "
          f"{math.degrees(command[1]):+10.1f}d {1:9d} "
          f"{np.linalg.norm(current[:2]-states[1][:2])*100:10.2f}")
    for tick in range(1, 4):
        before = current.copy()
        cmd = tracker.control(current, states, controls, offset=tick, neighbors=neighbour)
        for _ in range(5):
            current = bicycle_step(current, cmd, cfg.physics_dt, cfg)
        print(f"     {tick+1:5d} {f'({before[0]:.3f}, {before[1]:.3f})':>22} {cmd[0]:+11.3f} "
              f"{math.degrees(cmd[1]):+10.1f}d {tick+1:9d} "
              f"{np.linalg.norm(current[:2]-states[tick+1][:2])*100:10.2f}")

    rule("7.  0.2 s LATER: WHAT THE POLICY SEES NEXT")
    moved = np.linalg.norm(current[:2] - state[:2])
    new_distance = np.linalg.norm(goal - current[:2])
    print(f"  state x = [x={current[0]:.3f}, y={current[1]:.3f}, yaw={math.degrees(current[2]):.1f}deg, "
          f"v={current[3]:.3f}, steer={math.degrees(current[4]):.1f}deg]")
    print(f"  it travelled {moved:.3f} m; distance to goal {np.linalg.norm(goal - state[:2]):.3f} -> {new_distance:.3f} m")
    print(f"  time is now {elapsed + cfg.policy_dt:.1f} s, so {deadline - elapsed - cfg.policy_dt:.1f} s remain")
    print(f"  required average speed is now {new_distance / (deadline - elapsed - cfg.policy_dt):.3f} m/s "
          f"(was {np.linalg.norm(goal - state[:2]) / (deadline - elapsed):.3f})")
    print("\n  The whole 1.2 s reference is now DISCARDED. The policy reads this new state,")
    print("  emits three fresh numbers, and the cycle repeats. Nothing is ever planned")
    print("  further ahead than 1.2 s, which is why a different plant (MuJoCo, a slip")
    print("  model, the real car) changes almost nothing: the loop simply re-measures.")


if __name__ == "__main__":
    raise SystemExit(main())
