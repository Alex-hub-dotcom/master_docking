# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Sanity Check Script - Comprehensive Environment Diagnostics
================================================================

This script validates:
1. Robot spawn poses (active + static)
2. Goal/target position consistency
3. Connector sphere positions (male/female)
4. Arena boundaries
5. Motor functionality
6. Camera output
7. Collision detection
8. Basic RL episode

Run with:
    python teko_sanity_check.py --test all
    python teko_sanity_check.py --test geometry
    python teko_sanity_check.py --test motors
    python teko_sanity_check.py --test camera
    python teko_sanity_check.py --test episode

Author: Diagnostic tool for Alex's TEKO project
"""

import argparse
import sys
import time

# ==============================================================================
# STEP 0: Parse args BEFORE Isaac Sim starts
# ==============================================================================
parser = argparse.ArgumentParser(description="TEKO Sanity Check")
parser.add_argument("--test", type=str, default="all",
                    choices=["all", "geometry", "motors", "camera", "episode", "collision"],
                    help="Which test to run")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--save_images", action="store_true", help="Save camera images")
args = parser.parse_args()

# ==============================================================================
# STEP 1: Launch Isaac Sim
# ==============================================================================
from isaaclab.app import AppLauncher

launcher_args = argparse.Namespace(
    headless=args.headless,
    enable_cameras=True,
)
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

# ==============================================================================
# STEP 2: Imports (after Isaac Sim is running)
# ==============================================================================
import torch
import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, Sdf
from isaaclab.sim import SimulationContext

# Import TEKO environment
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_ok(msg: str):
    print(f"  ✅ {msg}")


def print_warn(msg: str):
    print(f"  ⚠️  {msg}")


def print_fail(msg: str):
    print(f"  ❌ {msg}")


def print_info(msg: str):
    print(f"  ℹ️  {msg}")


def create_debug_sphere(stage, name: str, position, color, radius=0.01):
    """Create a colored debug sphere at the given position."""
    path = f"/World/Debug/{name}"
    
    # Create sphere
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(radius)
    
    # Set position (convert to Python floats)
    xf = UsdGeom.Xformable(sphere)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    
    # Set color
    sphere.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    
    return path


def create_debug_box(stage, name: str, position, scale, color):
    """Create a colored debug box."""
    path = f"/World/Debug/{name}"
    
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xf.AddScaleOp().Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
    
    UsdGeom.Gprim(cube).CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    
    return path


# ==============================================================================
# TEST 1: GEOMETRY CHECK
# ==============================================================================

def test_geometry(env: TekoEnv):
    """Verify all geometric positions and relationships."""
    print_header("GEOMETRY CHECK")
    
    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/Debug", "Xform")
    
    issues = []
    
    # --- 1.1 Active Robot Position ---
    print("\n  [1.1] Active Robot Positions:")
    robot_pos = env.robot.data.root_pos_w
    robot_quat = env.robot.data.root_quat_w
    
    for i in range(min(env.num_envs, 4)):
        pos = robot_pos[i].cpu().numpy()
        quat = robot_quat[i].cpu().numpy()
        
        # Extract yaw from quaternion
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        yaw_rad = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_deg = np.rad2deg(yaw_rad)
        
        print(f"       Env {i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"yaw={yaw_deg:.1f}°")
        
        # Check Z height (should be ~0.4 for ground contact)
        if pos[2] < 0.35 or pos[2] > 0.45:
            issues.append(f"Env {i}: Robot Z={pos[2]:.3f} seems wrong (expected ~0.4)")
    
    # --- 1.2 Goal Positions ---
    print("\n  [1.2] Goal (Static Robot) Positions:")
    goal_pos = env.goal_positions
    
    for i in range(min(env.num_envs, 4)):
        pos = goal_pos[i].cpu().numpy()
        origin = env.scene.env_origins[i].cpu().numpy()
        local_pos = pos - origin
        
        print(f"       Env {i}: global=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"local=({local_pos[0]:.3f}, {local_pos[1]:.3f}, {local_pos[2]:.3f})")
        
        # Check consistency with config
        if hasattr(env.cfg, 'goal') and hasattr(env.cfg.goal, 'position'):
            cfg_pos = env.cfg.goal.position
            print(f"       Config says: {cfg_pos}")
            if abs(local_pos[0] - cfg_pos[0]) > 0.1:
                issues.append(f"Env {i}: Goal X mismatch! local={local_pos[0]:.2f}, cfg={cfg_pos[0]}")
    
    # --- 1.3 Connector Spheres ---
    print("\n  [1.3] Connector Sphere Positions:")
    female_pos, male_pos, surface_xy, surface_3d = env.get_sphere_distances_from_physics()
    
    for i in range(min(env.num_envs, 4)):
        f_pos = female_pos[i].cpu().numpy()
        m_pos = male_pos[i].cpu().numpy()
        dist_xy = surface_xy[i].item()
        dist_3d = surface_3d[i].item()
        
        print(f"       Env {i}: female=({f_pos[0]:.3f}, {f_pos[1]:.3f}, {f_pos[2]:.3f})")
        print(f"               male=({m_pos[0]:.3f}, {m_pos[1]:.3f}, {m_pos[2]:.3f})")
        print(f"               dist_xy={dist_xy:.4f}m, dist_3d={dist_3d:.4f}m")
        
        # Create visual markers (only for env 0)
        if i == 0:
            create_debug_sphere(stage, f"Female_{i}", f_pos.tolist(), (1.0, 0.0, 0.0), 0.008)
            create_debug_sphere(stage, f"Male_{i}", m_pos.tolist(), (0.0, 0.0, 1.0), 0.008)
    
    # --- 1.4 Arena Boundaries ---
    print("\n  [1.4] Arena Boundaries:")
    hx = env._arena_half_x
    hy = env._arena_half_y
    print(f"       half_x={hx:.2f}m, half_y={hy:.2f}m")
    print(f"       Arena spans: X=[{-hx:.2f}, {hx:.2f}], Y=[{-hy:.2f}, {hy:.2f}]")
    
    # Create boundary markers for env 0
    origin = env.scene.env_origins[0].cpu().numpy()
    z = 0.4
    create_debug_box(stage, "Boundary_Xmin", 
                     (origin[0] - hx, origin[1], z), (0.02, hy, 0.02), (1.0, 0.0, 0.0))
    create_debug_box(stage, "Boundary_Xmax", 
                     (origin[0] + hx, origin[1], z), (0.02, hy, 0.02), (1.0, 0.0, 0.0))
    create_debug_box(stage, "Boundary_Ymin", 
                     (origin[0], origin[1] - hy, z), (hx, 0.02, 0.02), (0.0, 1.0, 0.0))
    create_debug_box(stage, "Boundary_Ymax", 
                     (origin[0], origin[1] + hy, z), (hx, 0.02, 0.02), (0.0, 1.0, 0.0))
    
    # --- 1.5 Body Dimensions ---
    print("\n  [1.5] Body Footprints:")
    print(f"       Active robot: {env._active_body_length:.3f}m x {env._active_body_width:.3f}m")
    print(f"       Static robot: {env._static_body_length:.3f}m x {env._static_body_width:.3f}m")
    
    # --- Summary ---
    print("\n  [SUMMARY]")
    if issues:
        for issue in issues:
            print_warn(issue)
    else:
        print_ok("All geometry checks passed!")
    
    return len(issues) == 0


# ==============================================================================
# TEST 2: MOTOR CHECK
# ==============================================================================

def test_motors(env: TekoEnv):
    """Test motor functionality with simple commands."""
    print_header("MOTOR CHECK")
    
    issues = []
    
    # --- 2.1 DOF Info ---
    print("\n  [2.1] DOF Configuration:")
    print(f"       Joint names: {env.robot.joint_names}")
    print(f"       DOF indices: {env.dof_idx}")
    print(f"       Max torque: {env._max_wheel_torque}")
    print(f"       Wheel polarity: {env.cfg.wheel_polarity}")
    
    # --- 2.2 Forward Motion Test ---
    print("\n  [2.2] Forward Motion Test:")
    
    # Get initial position
    env.reset()
    initial_pos = env.robot.data.root_pos_w.clone()
    
    # Apply forward action for 50 steps
    forward_action = torch.tensor([[1.0, 0.0]], device=env.device).expand(env.num_envs, -1)
    
    for _ in range(50):
        env.step(forward_action)
    
    final_pos = env.robot.data.root_pos_w
    displacement = final_pos - initial_pos
    
    for i in range(min(env.num_envs, 2)):
        dx = displacement[i, 0].item()
        dy = displacement[i, 1].item()
        print(f"       Env {i}: moved dx={dx:.4f}m, dy={dy:.4f}m")
        
        # Robot faces backward (yaw=π), so forward should move in -X
        if dx > 0.01:
            issues.append(f"Env {i}: Forward action moved in +X (expected -X)")
        elif abs(dx) < 0.001:
            issues.append(f"Env {i}: No movement detected!")
    
    # --- 2.3 Rotation Test ---
    print("\n  [2.3] Rotation Test:")
    
    env.reset()
    initial_quat = env.robot.data.root_quat_w.clone()
    
    # Apply rotation action
    rotate_action = torch.tensor([[0.0, 1.0]], device=env.device).expand(env.num_envs, -1)
    
    for _ in range(50):
        env.step(rotate_action)
    
    final_quat = env.robot.data.root_quat_w
    
    for i in range(min(env.num_envs, 2)):
        # Extract yaw change
        def quat_to_yaw(q):
            w, x, y, z = q[0], q[1], q[2], q[3]
            return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        init_yaw = quat_to_yaw(initial_quat[i].cpu().numpy())
        final_yaw = quat_to_yaw(final_quat[i].cpu().numpy())
        delta_yaw = np.rad2deg(final_yaw - init_yaw)
        
        print(f"       Env {i}: rotated {delta_yaw:.2f}°")
        
        if abs(delta_yaw) < 1.0:
            issues.append(f"Env {i}: Rotation command had no effect!")
    
    # --- Summary ---
    print("\n  [SUMMARY]")
    if issues:
        for issue in issues:
            print_warn(issue)
    else:
        print_ok("All motor checks passed!")
    
    return len(issues) == 0


# ==============================================================================
# TEST 3: CAMERA CHECK
# ==============================================================================

def test_camera(env: TekoEnv, save_images: bool = False):
    """Test camera functionality."""
    print_header("CAMERA CHECK")
    
    issues = []
    
    # --- 3.1 Camera Info ---
    print("\n  [3.1] Camera Configuration:")
    print(f"       Resolution: {env.cfg.camera.width}x{env.cfg.camera.height}")
    print(f"       Num cameras: {len(env.cameras)}")
    print(f"       Frame stack: {env.num_frame_stack}")
    
    # --- 3.2 Get Observations ---
    print("\n  [3.2] Observation Test:")
    
    env.reset()
    
    # Step to get camera data
    for _ in range(5):
        action = torch.zeros((env.num_envs, 2), device=env.device)
        env.step(action)
    
    obs = env._get_observations()
    
    if "rgb" in obs:
        rgb = obs["rgb"]
        print(f"       Shape: {rgb.shape}")
        print(f"       Dtype: {rgb.dtype}")
        print(f"       Range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
        print(f"       Mean: {rgb.mean().item():.3f}")
        print(f"       Std: {rgb.std().item():.3f}")
        
        # Check for issues
        if rgb.max().item() < 0.01:
            issues.append("Image is all black!")
        if rgb.std().item() < 0.01:
            issues.append("Image has no variation (constant color)")
        if torch.isnan(rgb).any():
            issues.append("Image contains NaN values!")
        
        # Save sample image
        if save_images:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # Get first frame from first environment
                img = rgb[0, 0].cpu().numpy()  # First env, first frame
                
                plt.figure(figsize=(8, 8))
                plt.imshow(img, cmap='gray', vmin=0, vmax=1)
                plt.title(f"Camera Frame (shape={img.shape})")
                plt.colorbar()
                plt.savefig("/workspace/teko/camera_test.png", dpi=150)
                plt.close()
                print_ok("Saved camera image to /workspace/teko/camera_test.png")
            except Exception as e:
                print_warn(f"Could not save image: {e}")
    else:
        issues.append("No 'rgb' key in observations!")
        print(f"       Available keys: {obs.keys()}")
    
    # --- Summary ---
    print("\n  [SUMMARY]")
    if issues:
        for issue in issues:
            print_warn(issue)
    else:
        print_ok("All camera checks passed!")
    
    return len(issues) == 0


# ==============================================================================
# TEST 4: COLLISION CHECK
# ==============================================================================

def test_collision(env: TekoEnv):
    """Test collision detection logic."""
    print_header("COLLISION CHECK")
    
    issues = []
    
    # --- 4.1 Initial State (no collision) ---
    print("\n  [4.1] Initial State (should be no collision):")
    
    env.reset()
    
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    
    diff = robot_pos - goal_pos
    dx = diff[:, 0].abs()
    dy = diff[:, 1].abs()
    
    sL = 0.5 * env._static_body_length
    sW = 0.5 * env._static_body_width
    aL = 0.5 * env._active_body_length
    aW = 0.5 * env._active_body_width
    
    overlap_x = dx < (sL + aL)
    overlap_y = dy < (sW + aW)
    overlap = overlap_x & overlap_y
    
    print(f"       dx range: [{dx.min().item():.3f}, {dx.max().item():.3f}]")
    print(f"       dy range: [{dy.min().item():.3f}, {dy.max().item():.3f}]")
    print(f"       Threshold: dx < {sL + aL:.3f}, dy < {sW + aW:.3f}")
    print(f"       Overlapping envs: {overlap.sum().item()}/{env.num_envs}")
    
    if overlap.any():
        issues.append(f"{overlap.sum().item()} environments start in collision!")
    
    # --- 4.2 Simulate approach ---
    print("\n  [4.2] Approach Simulation:")
    
    # Move robot toward goal
    approach_action = torch.tensor([[-0.5, 0.0]], device=env.device).expand(env.num_envs, -1)
    
    collision_detected = False
    for step in range(100):
        obs, reward, term, trunc, info = env.step(approach_action)
        
        if term.any():
            print(f"       Termination at step {step}: {term.sum().item()} envs")
            collision_detected = True
            break
    
    if not collision_detected:
        print_info("No collision detected in 100 steps (robot may not have reached goal)")
    
    # --- Summary ---
    print("\n  [SUMMARY]")
    if issues:
        for issue in issues:
            print_warn(issue)
    else:
        print_ok("Collision check logic seems OK")
    
    return len(issues) == 0


# ==============================================================================
# TEST 5: FULL EPISODE CHECK
# ==============================================================================

def test_episode(env: TekoEnv):
    """Run a full episode and check rewards/termination."""
    print_header("EPISODE CHECK")
    
    issues = []
    
    # --- 5.1 Reset ---
    print("\n  [5.1] Episode Reset:")
    obs, info = env.reset()
    
    print(f"       Observation keys: {obs.keys()}")
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            print(f"       {k}: shape={v.shape}, dtype={v.dtype}")
    
    # --- 5.2 Run Episode ---
    print("\n  [5.2] Running Episode (max 300 steps):")
    
    total_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    successes = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    for step in range(300):
        # Random action
        action = torch.randn((env.num_envs, 2), device=env.device) * 0.5
        
        obs, reward, term, trunc, info = env.step(action)
        
        total_rewards += reward
        episode_lengths += 1
        
        # Track terminations
        done = term | trunc
        if done.any():
            # Check for successes (high reward on termination)
            successes |= (reward > 50) & done
        
        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
            print(f"       Step {step+1}: mean_reward={reward.mean().item():.3f}, "
                  f"mean_dist={surface_xy.mean().item():.4f}m, "
                  f"done={done.sum().item()}")
    
    # --- 5.3 Results ---
    print("\n  [5.3] Episode Results:")
    print(f"       Mean total reward: {total_rewards.mean().item():.2f}")
    print(f"       Mean episode length: {episode_lengths.float().mean().item():.1f}")
    print(f"       Successes: {successes.sum().item()}/{env.num_envs}")
    
    # Check for issues
    if total_rewards.mean().item() < -100:
        issues.append("Very negative rewards - check reward function!")
    if total_rewards.std().item() < 0.1:
        issues.append("No reward variation - rewards may not be computed correctly")
    
    # --- Summary ---
    print("\n  [SUMMARY]")
    if issues:
        for issue in issues:
            print_warn(issue)
    else:
        print_ok("Episode check passed!")
    
    return len(issues) == 0


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 70)
    print("  TEKO SANITY CHECK - Comprehensive Environment Diagnostics")
    print("=" * 70)
    print(f"  Test: {args.test}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Headless: {args.headless}")
    print("=" * 70)
    
    # Create environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_boundaries = True
    cfg.debug_robot_boxes = True
    
    env = None
    all_passed = True
    
    try:
        print("\n[INFO] Creating environment...")
        env = TekoEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        
        # Reset to initialize
        env.reset()
        
        # Let physics settle
        print("[INFO] Waiting for physics to settle...")
        for _ in range(30):
            action = torch.zeros((args.num_envs, 2), device=env.device)
            env.step(action)
        
        # Run tests
        if args.test in ["all", "geometry"]:
            if not test_geometry(env):
                all_passed = False
        
        if args.test in ["all", "motors"]:
            if not test_motors(env):
                all_passed = False
        
        if args.test in ["all", "camera"]:
            if not test_camera(env, save_images=args.save_images):
                all_passed = False
        
        if args.test in ["all", "collision"]:
            if not test_collision(env):
                all_passed = False
        
        if args.test in ["all", "episode"]:
            if not test_episode(env):
                all_passed = False
        
        # Final summary
        print_header("FINAL SUMMARY")
        if all_passed:
            print_ok("ALL TESTS PASSED!")
        else:
            print_fail("SOME TESTS FAILED - Review warnings above")
        
        # Keep running if not headless
        if not args.headless:
            print("\n[INFO] Press Ctrl+C to exit, or view in Isaac Sim GUI...")
            while simulation_app.is_running():
                simulation_app.update()
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())