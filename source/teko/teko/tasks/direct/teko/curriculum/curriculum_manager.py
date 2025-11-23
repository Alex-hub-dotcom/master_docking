# SPDX-License-Identifier: BSD-3-Clause
"""
16-STAGE ULTRA-SMOOTH CURRICULUM FOR TEKO (VISION-ONLY, OFFSET-FOCUSED)
=======================================================================

This file defines how the TEKO robot is spawned for each curriculum stage.
Only the *initial pose* (position + yaw) is changed per stage; rewards,
actions, etc. are defined elsewhere.

Conventions:
- yaw = π  -> "dock-ready" orientation (rear side / camera facing the goal)
- yaw = 0  -> robot turned 180° away from dock (needs to turn around)

High-level structure:
- Stages 0–3  : Forward-only, increasing distance.
- Stages 4–11 : Short/medium distance (0.20–0.40 m), increasing lateral + yaw offset.
- Stages 12–13: 180° cases (facing away from goal), close distance.
- Stages 14–15: Search / full autonomy in the arena.

Distance design (vision-only friendly):
- For offset stages (4–11) we deliberately keep distances capped at 0.40 m
  so that the static robot remains clearly visible even without ArUco markers.
"""

import numpy as np
import torch

from ..utils.geometry_utils import yaw_to_quat


# Descriptive names for console logging
STAGE_NAMES = [
    "Stage 0:  Baby Steps (5–12 cm, forward)",
    "Stage 1:  Forward 1 (10–18 cm, forward)",
    "Stage 2:  Forward 2 (15–25 cm, forward)",
    "Stage 3:  Medium Forward (20–35 cm, forward)",
    "Stage 4:  Tiny Offset Close (20–30 cm, ±3°, ±3 cm)",
    "Stage 5:  Tiny Offset Medium (20–40 cm, ±6°, ±5 cm)",
    "Stage 6:  Small Offset (20–40 cm, ±9°, ±7 cm)",
    "Stage 7:  Small+ Offset (20–40 cm, ±12°, ±9 cm)",
    "Stage 8:  Medium Offset (20–40 cm, ±15°, ±11 cm)",
    "Stage 9:  Medium+ Offset (20–40 cm, ±18°, ±13 cm)",
    "Stage 10: Large Offset (20–40 cm, ±21°, ±16 cm)",
    "Stage 11: Large+ Offset (20–40 cm, ±24°, ±18 cm)",
    "Stage 12: 180° Close (25–40 cm, turn around)",
    "Stage 13: 180° Offset (25–40 cm, 0°±10°, ±10 cm)",
    "Stage 14: Arena Search (0.60–1.20 m, random yaw)",
    "Stage 15: Full Autonomy (random in arena, random yaw)",
]


# =============================================================================
# Dispatcher
# =============================================================================

def reset_environment_curriculum(env, env_ids):
    """
    Reset environments according to the current curriculum stage, with replay
    from the previous stage to prevent catastrophic forgetting.

    Replay policy:
    - Base: 20% of envs sample from the previous stage
    - Hard stages (>= 7): 30%
    - Very hard stages (>= 10): 40%
    """
    current_stage = int(env.curriculum_level)
    num = len(env_ids)

    # Stage 0 has no previous stage
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
    """Route to the correct _reset_stageX function."""
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
    Helper for pure forward docking stages.

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
    min_dist: float,
    max_dist: float,
):
    """
    Helper for lateral-offset stages around yaw ≈ π (rear facing the goal).

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

# Stage 0: Baby Steps (0.05–0.12 m, forward)
def _reset_stage0(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.05, 0.12, yaw)


# Stage 1: Forward 1 (0.10–0.18 m, forward)
def _reset_stage1(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.10, 0.18, yaw)


# Stage 2: Forward 2 (0.15–0.25 m, forward)
def _reset_stage2(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.15, 0.25, yaw)


# Stage 3: Medium Forward (0.20–0.35 m, forward)
def _reset_stage3(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.20, 0.35, yaw)


# Stage 4: Tiny Offset Close (20–30 cm, ±3°, ±3 cm)
def _reset_stage4(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=3.0,
        lateral_m=0.03,
        min_dist=0.20,
        max_dist=0.30,
    )


# Stage 5: Tiny Offset Medium (20–40 cm, ±6°, ±5 cm)
def _reset_stage5(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=6.0,
        lateral_m=0.05,
        min_dist=0.20,
        max_dist=0.40,
    )


# Stage 6: Small Offset (20–40 cm, ±9°, ±7 cm)
def _reset_stage6(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=9.0,
        lateral_m=0.07,
        min_dist=0.20,
        max_dist=0.40,
    )


# Stage 7: Small+ Offset (20–40 cm, ±12°, ±9 cm)
def _reset_stage7(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=12.0,
        lateral_m=0.09,
        min_dist=0.20,
        max_dist=0.40,
    )


# Stage 8: Medium Offset (20–40 cm, ±15°, ±11 cm)
def _reset_stage8(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=15.0,
        lateral_m=0.11,
        min_dist=0.20,
        max_dist=0.40,
    )


# Stage 9: Medium+ Offset (20–40 cm, ±18°, ±13 cm)
def _reset_stage9(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=18.0,
        lateral_m=0.13,
        min_dist=0.20,
        max_dist=0.40,
    )


# Stage 10: Large Offset (20–40 cm, ±21°, ±16 cm)
def _reset_stage10(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=21.0,
        lateral_m=0.16,
        min_dist=0.20,
        max_dist=0.40,
    )


# Stage 11: Large+ Offset (20–40 cm, ±24°, ±18 cm)
def _reset_stage11(env, env_ids):
    _offset_reset(
        env,
        env_ids,
        angle_deg=24.0,
        lateral_m=0.18,
        min_dist=0.20,
        max_dist=0.40,
    )


# Stage 12: 180° Close (25–40 cm, facing away, 0 offset)
def _reset_stage12(env, env_ids):
    """
    Robot starts close but facing AWAY from the goal (needs to turn ~180°).
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.15 + 0.25  # 0.25–0.40 m

    yaw = torch.zeros(num, device=env.device)  # front towards goal, rear away

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 13: 180° Offset (25–40 cm, 0°±10°, ±10 cm)
def _reset_stage13(env, env_ids):
    """
    Robot starts close, facing away with a small yaw + lateral offset.
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.15 + 0.25  # 0.25–0.40 m

    max_yaw = np.deg2rad(10.0)
    yaw = torch.rand(num, device=env.device) * (2 * max_yaw) - max_yaw  # 0° ± 10°

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (
        torch.rand(num, device=env.device) * 0.20 - 0.10
    )
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 14: Arena Search (0.60–1.20 m, random yaw)
def _reset_stage14(env, env_ids):
    """
    Robot starts farther away and may need to search for the goal.
    Distances are limited to 0.60–1.20 m to keep the goal still visible
    in the camera without ArUco markers.
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.60 + 0.60  # 0.60–1.20 m
    yaw = torch.rand(num, device=env.device) * 2 * np.pi     # fully random yaw

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (
        torch.rand(num, device=env.device) * 0.60 - 0.30
    )
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 15: Full Autonomy (random position in arena, random yaw)
def _reset_stage15(env, env_ids):
    """
    Robot spawns anywhere inside the logical arena, with random yaw.
    This approximates a production-like setup, but always stays
    within the red boundary walls.
    """
    num = len(env_ids)
    device = env.device

    # Use the arena half extents from the environment
    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)

    # Small margins so we do not spawn exactly on top of the walls
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
# Curriculum control
# =============================================================================

def set_curriculum_level(env, level: int):
    """
    Set the curriculum stage (0–15) on the environment and print a log line.
    """
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level

    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """
    Decide whether to advance to the next curriculum stage based on success rate.
    The trainer additionally enforces a minimum number of steps per stage.
    """
    max_level = len(STAGE_NAMES) - 1
    if current_level >= max_level:
        return False

    # Demand a high success rate for robustness
    return success_rate >= 0.85
