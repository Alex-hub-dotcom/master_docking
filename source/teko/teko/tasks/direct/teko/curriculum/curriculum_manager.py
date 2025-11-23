# SPDX-License-Identifier: BSD-3-Clause
"""
16-STAGE ULTRA-SMOOTH CURRICULUM FOR TEKO
=========================================

This file defines how the TEKO robot is spawned for each curriculum stage.
Only the *initial pose* (position + yaw) is changed per stage; rewards,
actions, etc. are defined elsewhere.

Important convention:
- yaw = π  -> "dock-ready" orientation (rear side / camera facing the goal)
- yaw = 0  -> robot turned 180° away from dock (needs to turn around)

Forward-only stages (no lateral offset, yaw ≈ π):
- Stage 0:  Baby Steps        (0.05–0.15 m)
- Stage 1:  Forward 1         (0.10–0.20 m)
- Stage 2:  Forward 2         (0.15–0.25 m)
- Stage 3:  Medium Forward    (0.20–0.35 m)
- Stage 4:  Medium+ Forward   (0.30–0.50 m)

Offset stages around yaw ≈ π (smoother progression):
- Stage 5:  Tiny Offset       (0.25–0.40 m, π±3°,   ±3 cm)
- Stage 6:  Small Offset      (0.25–0.40 m, π±6°,   ±6 cm)
- Stage 7:  Small+ Offset     (0.30–0.50 m, π±8°,   ±8 cm)
- Stage 8:  Medium Offset     (0.30–0.50 m, π±10°,  ±10 cm)
- Stage 9:  Medium+ Offset    (0.30–0.50 m, π±13°,  ±13 cm)
- Stage 10: Large Offset      (0.30–0.50 m, π±15°,  ±15 cm)
- Stage 11: Large+ Offset     (0.30–0.50 m, π±18°,  ±18 cm)

180° and search:
- Stage 12: 180° Close        (0.30–0.50 m, yaw ≈ 0,      0 cm)
- Stage 13: 180° Offset       (0.30–0.50 m, 0°±10°,      ±10 cm)
- Stage 14: Arena Search      (0.80–1.50 m, random yaw)
- Stage 15: Full Autonomy     (random position, random yaw)

Advancement:
- Logic to move to the next stage is implemented in the trainer
  (e.g., success rate >= threshold + minimum steps per stage).
"""

import numpy as np
import torch

from ..utils.geometry_utils import yaw_to_quat


# Descriptive names for pretty logging
STAGE_NAMES = [
    "Stage 0:  Baby Steps (5–15 cm)",
    "Stage 1:  Forward 1 (10–20 cm)",
    "Stage 2:  Forward 2 (15–25 cm)",
    "Stage 3:  Medium Forward (20–35 cm)",
    "Stage 4:  Medium+ Forward (30–50 cm)",
    "Stage 5:  Tiny Offset (±3°, ±3 cm)",
    "Stage 6:  Small Offset (±6°, ±6 cm)",
    "Stage 7:  Small+ Offset (±8°, ±8 cm)",
    "Stage 8:  Medium Offset (±10°, ±10 cm)",
    "Stage 9:  Medium+ Offset (±13°, ±13 cm)",
    "Stage 10: Large Offset (±15°, ±15 cm)",
    "Stage 11: Large+ Offset (±18°, ±18 cm)",
    "Stage 12: 180° Close (turn around)",
    "Stage 13: 180° Offset (turn + align)",
    "Stage 14: Arena Search (far + random)",
    "Stage 15: Full Autonomy (production)",
]


# =============================================================================
# Dispatcher
# =============================================================================

def reset_environment_curriculum(env, env_ids):
    """
    Reset with replay from previous stage to prevent forgetting.

    - Base: 20% of envs sample from previous stage
    - Harder stages (>= 7): 30%
    - Very hard stages (>= 10): 40%
    """
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    # Stage 0: no previous stage exists
    if current_stage == 0:
        _reset_stage_dispatch(env, env_ids, current_stage)
        return

    device = env.device

    # Stage-dependent previous-stage replay probability
    mix_prob = 0.2
    if current_stage >= 10:
        mix_prob = 0.4
    elif current_stage >= 7:
        mix_prob = 0.3

    mix_prev = torch.rand(num, device=device) < mix_prob

    prev_ids = env_ids[mix_prev]
    curr_ids = env_ids[~mix_prev]

    if len(prev_ids) > 0:
        _reset_stage_dispatch(env, prev_ids, current_stage - 1)

    if len(curr_ids) > 0:
        _reset_stage_dispatch(env, curr_ids, current_stage)


def _reset_stage_dispatch(env, env_ids, stage: int):
    """Small helper to route to the correct _reset_stageX function."""
    if stage == 0:
        _reset_stage0(env, env_ids)
    elif stage == 1:
        _reset_stage1(env, env_ids)
    elif stage == 2:
        _reset_stage2(env, env_ids)
    elif stage == 3:
        _reset_stage3(env, env_ids)
    elif stage == 4:
        _reset_stage4(env, env_ids)
    elif stage == 5:
        _reset_stage5(env, env_ids)
    elif stage == 6:
        _reset_stage6(env, env_ids)
    elif stage == 7:
        _reset_stage7(env, env_ids)
    elif stage == 8:
        _reset_stage8(env, env_ids)
    elif stage == 9:
        _reset_stage9(env, env_ids)
    elif stage == 10:
        _reset_stage10(env, env_ids)
    elif stage == 11:
        _reset_stage11(env, env_ids)
    elif stage == 12:
        _reset_stage12(env, env_ids)
    elif stage == 13:
        _reset_stage13(env, env_ids)
    elif stage == 14:
        _reset_stage14(env, env_ids)
    elif stage == 15:
        _reset_stage15(env, env_ids)
    else:
        raise ValueError(f"Invalid curriculum stage: {stage}")


# =============================================================================
# Helpers
# =============================================================================

def _base_forward_reset(env, env_ids, min_dist: float, max_dist: float, yaw: torch.Tensor):
    """
    Helper for "forward docking" stages.

    Places the robot directly in front of the goal, at a random distance
    in [min_dist, max_dist], with a fixed yaw.

    Convention:
    - yaw ≈ π  -> rear side (camera) faces the goal (dock-ready)
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * (max_dist - min_dist) + min_dist

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40  # TEKO root height

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


def _offset_reset(
    env,
    env_ids,
    angle_deg: float,
    lateral_m: float,
    min_dist: float = 0.30,
    max_dist: float = 0.50,
):
    """
    Helper for lateral-offset stages around yaw ≈ π (rear facing goal).

    - Distance: [min_dist, max_dist]
    - Yaw:      π ± angle_deg
    - Lateral:  ± lateral_m in Y
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * (max_dist - min_dist) + min_dist

    max_yaw = np.deg2rad(angle_deg)
    yaw = np.pi + (torch.rand(num, device=env.device) * (2 * max_yaw) - max_yaw)

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (
        torch.rand(num, device=env.device) * (2 * lateral_m) - lateral_m
    )
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# Stage implementations
# =============================================================================

# Stage 0: Baby Steps (0.05–0.15 m)
def _reset_stage0(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.05, 0.15, yaw)


# Stage 1: Forward 1 (0.10–0.20 m)
def _reset_stage1(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.10, 0.20, yaw)


# Stage 2: Forward 2 (0.15–0.25 m)
def _reset_stage2(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.15, 0.25, yaw)


# Stage 3: Medium Forward (0.20–0.35 m)
def _reset_stage3(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.20, 0.35, yaw)


# Stage 4: Medium+ Forward (0.30–0.50 m)
def _reset_stage4(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.30, 0.50, yaw)


# Stage 5: Tiny Offset (±3°, ±3 cm, 0.25–0.40 m)
def _reset_stage5(env, env_ids):
    _offset_reset(env, env_ids,
                  angle_deg=3.0,
                  lateral_m=0.03,
                  min_dist=0.25,
                  max_dist=0.40)


# Stage 6: Small Offset (±6°, ±6 cm, 0.25–0.40 m)
def _reset_stage6(env, env_ids):
    _offset_reset(env, env_ids,
                  angle_deg=6.0,
                  lateral_m=0.06,
                  min_dist=0.25,
                  max_dist=0.40)


# Stage 7: Small+ Offset (±8°, ±8 cm, 0.30–0.50 m)
def _reset_stage7(env, env_ids):
    _offset_reset(env, env_ids,
                  angle_deg=8.0,
                  lateral_m=0.08,
                  min_dist=0.30,
                  max_dist=0.50)


# Stage 8: Medium Offset (±10°, ±10 cm)
def _reset_stage8(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=10.0, lateral_m=0.10)


# Stage 9: Medium+ Offset (±13°, ±13 cm)
def _reset_stage9(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=13.0, lateral_m=0.13)


# Stage 10: Large Offset (±15°, ±15 cm)
def _reset_stage10(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=15.0, lateral_m=0.15)


# Stage 11: Large+ Offset (±18°, ±18 cm)
def _reset_stage11(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=18.0, lateral_m=0.18)


# Stage 12: 180° Close (facing away, 0 offset)
def _reset_stage12(env, env_ids):
    """
    Robot starts close but facing AWAY from the goal (needs to turn ~180°).
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30  # 0.30–0.50 m

    yaw = torch.zeros(num, device=env.device)  # front towards goal, rear away

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 13: 180° Offset (facing away ±10°, ±10 cm)
def _reset_stage13(env, env_ids):
    """
    Robot starts close, facing away with a small yaw + lateral offset.
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30  # 0.30–0.50 m

    max_yaw = np.deg2rad(10.0)
    yaw = torch.rand(num, device=env.device) * (2 * max_yaw) - max_yaw  # 0° ± 10°

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.20 - 0.10)
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 14: Arena Search (0.80–1.50 m, random yaw)
def _reset_stage14(env, env_ids):
    """
    Robot starts farther away and may need to search for the goal.
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.70 + 0.80  # 0.80–1.50 m
    yaw = torch.rand(num, device=env.device) * 2 * np.pi     # fully random

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.60 - 0.30)
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 15: Full Autonomy (random in arena)
def _reset_stage15(env, env_ids):
    """
    Robot spawns anywhere inside the logical arena, random yaw.
    This approximates a production-like setup, but always stays
    within the red boundary walls.
    """
    num = len(env_ids)
    device = env.device

    # Use the arena half-extents from the env
    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)

    # Small margin so we don't spawn exactly on the walls
    margin_x = 0.1
    margin_y = 0.1

    # Sample local (env-frame) positions inside the arena
    x_local = torch.rand(num, device=device) * (2 * (hx - margin_x)) - (hx - margin_x)
    y_local = torch.rand(num, device=device) * (2 * (hy - margin_y)) - (hy - margin_y)

    # Convert env-local positions to world using env origins
    env_origins = env.scene.env_origins[env_ids]  # [N, 3]
    x = env_origins[:, 0] + x_local
    y = env_origins[:, 1] + y_local
    z = torch.ones(num, device=device) * 0.40  # TEKO root height

    yaw = torch.rand(num, device=device) * 2 * np.pi

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# Curriculum Control
# =============================================================================

def set_curriculum_level(env, level: int):
    """
    Set the curriculum stage (0–15) on the environment and print a nice log.
    """
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level

    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """
    Decide whether to advance to the next curriculum stage.

    NOTE:
    - This function *only* checks the success rate.
    - The trainer enforces a minimum number of steps per stage.
    """
    max_level = len(STAGE_NAMES) - 1
    if current_level >= max_level:
        return False

    return success_rate >= 0.85
