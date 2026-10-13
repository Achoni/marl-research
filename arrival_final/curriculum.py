"""Training distributions; progress uses the saved curriculum budget.

Validation always uses the requested final configuration, not these stages.
Short smoke runs do not rush through the curriculum: ``--steps`` is independent
of ``--curriculum-steps``. The trainer changes stages only between PPO rollouts.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .config import EnvConfig

STAGE_BOUNDARIES = (0.35, 0.60, 0.80)
STAGE_NAMES = ("converging", "pacing", "interactions", "full")


def stage_index(step: int, curriculum_steps: int, mode: str = "staged") -> int:
    if curriculum_steps < 1 or step < 0:
        raise ValueError("step must be nonnegative and curriculum_steps positive")
    if mode == "none":
        return 3
    if mode != "staged":
        raise ValueError("curriculum must be staged or none")
    return sum(step >= math.ceil(fraction * curriculum_steps)
               for fraction in STAGE_BOUNDARIES)


def next_boundary(step: int, curriculum_steps: int, mode: str = "staged"):
    """Return the next rollout cut, or None after the last distribution change."""
    stage_index(step, curriculum_steps, mode)
    if mode == "none":
        return None
    for fraction in STAGE_BOUNDARIES:
        boundary = math.ceil(fraction * curriculum_steps)
        if boundary > step:
            return boundary
    return None


def training_config(target: EnvConfig, step: int, curriculum_steps: int,
                    mode: str = "staged") -> tuple[str, EnvConfig]:
    """Override only scenario difficulty; controller/reward/physics stay fixed."""
    index = stage_index(step, curriculum_steps, mode)
    if index == 0:
        changes = dict(family="converging", family_weights=None,
                       arena_half_min=4.5, arena_half_max=5.0,
                       heading_jitter=0.0, initial_speed_max=0.0,
                       deadline_slack_min=1.5, deadline_slack_max=1.5)
    elif index == 1:
        changes = dict(family="mixed",
                       family_weights={"converging": 0.45, "unequal": 0.35,
                                       "parallel": 0.20},
                       arena_half_min=4.5, arena_half_max=5.0,
                       heading_jitter=0.0, initial_speed_max=0.0,
                       deadline_slack_min=1.5, deadline_slack_max=1.8)
    elif index == 2:
        changes = dict(family="mixed",
                       family_weights={"converging": 0.25, "unequal": 0.20,
                                       "crossing": 0.25, "random": 0.25,
                                       "parallel": 0.05},
                       arena_half_min=4.5, arena_half_max=6.0,
                       heading_jitter=min(target.heading_jitter, math.radians(20)),
                       initial_speed_max=min(target.initial_speed_max, 0.2),
                       deadline_slack_min=1.5, deadline_slack_max=2.0)
    else:
        changes = {}
    # A family the target distribution has zeroed must not reappear in a curriculum stage.
    allowed = ({name for name, weight in target.family_weights.items() if weight > 0}
               if target.family_weights is not None else
               {"random", "crossing", "converging", "parallel", "unequal"})
    if target.family != "mixed":
        allowed = {target.family}
    dropped = {"random", "crossing", "converging", "parallel", "unequal"} - allowed
    if dropped and "family_weights" in changes and changes["family_weights"]:
        kept = {name: weight for name, weight in changes["family_weights"].items()
                if name not in dropped}
        changes["family_weights"] = kept or target.family_weights
        if not kept:
            changes["family"] = target.family
    if changes.get("family") in dropped:
        changes = {}
    return STAGE_NAMES[index], replace(target, **changes).validate()
