#!/usr/bin/env python3
"""
Sanity check V3: Uses USD API directly to get prim positions.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def maybe_add_teko_to_syspath() -> None:
    for p in ("/workspace/teko/source/teko", "/home/schux00/teko/source/teko"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return


def get_world_position_usd(stage, prim_path: str):
    """Get world position using USD Xform cache."""
    from pxr import UsdGeom, Gf
    
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] Prim not found: {prim_path}")
        return None
    
    xform_cache = UsdGeom.XformCache()
    world_transform = xform_cache.GetLocalToWorldTransform(prim)
    translation = world_transform.ExtractTranslation()
    
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def create_debug_sphere(stage, prim_path: str, pos_xyz, radius: float, rgb):
    """Create a colored debug sphere."""
    from pxr import UsdGeom, Gf, Sdf
    
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        sph = UsdGeom.Sphere.Define(stage, Sdf.Path(prim_path))
        sph.CreateRadiusAttr(float(radius))
        UsdGeom.Gprim(sph).CreateDisplayColorAttr(
            [Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))]
        )
        xf = UsdGeom.Xformable(sph)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))
    else:
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))
                return
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))


def rotate_xy(dx, dy, deg):
    a = math.radians(float(deg))
    ca, sa = math.cos(a), math.sin(a)
    return (ca * dx - sa * dy, sa * dx + ca * dy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, default="vision", choices=["vision", "state"])
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="/workspace/teko/scripts/sanity_checks/out/goal")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--active_robot_root_path", type=str, default="/World/envs/env_0/Robot")
    parser.add_argument("--female_sphere_prim", type=str, 
                        default="/World/envs/env_0/Robot/teko_urdf/TEKO_Body/TEKO_ConnectorRear/SphereRear")
    parser.add_argument("--male_sphere_prim", type=str,
                        default="/World/envs/env_0/RobotGoal/teko_urdf/TEKO_Body/TEKO_ConnectorMale/TEKO_ConnectorPin/SpherePin")
    parser.add_argument("--teleport", type=int, default=1)
    parser.add_argument("--lookat_z", type=float, default=0.35)
    parser.add_argument("--cam_back", type=float, default=0.0)
    parser.add_argument("--cam_side", type=float, default=-2.5)
    parser.add_argument("--cam_up", type=float, default=1.2)
    parser.add_argument("--cam_yaw_deg", type=float, default=0.0)
    parser.add_argument("--stage_res_w", type=int, default=1280)
    parser.add_argument("--stage_res_h", type=int, default=720)
    parser.add_argument("--debug_radius", type=float, default=0.02)
    parser.add_argument("--debug_lift", type=float, default=0.02)

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    
    app = AppLauncher(args)
    sim_app = app.app

    maybe_add_teko_to_syspath()

    import torch
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom, Gf

    # Build env
    if args.env_type == "vision":
        from teko.tasks.direct.teko.teko_env import TekoEnv
        from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
        cfg = TekoEnvCfg()
        if hasattr(cfg, "enable_curriculum"):
            cfg.enable_curriculum = False
        cfg.scene.num_envs = args.num_envs
        env = TekoEnv(cfg=cfg)
    else:
        from teko.tasks.direct.teko.teko_env_state import TekoEnvState
        from teko.tasks.direct.teko.teko_env_cfg_state import TekoEnvCfgState
        cfg = TekoEnvCfgState()
        cfg.scene.num_envs = args.num_envs
        env = TekoEnvState(cfg=cfg)

    out_root = Path(args.output_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{ts}"
    stage_out = run_dir / "stage_camera"
    stage_out.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        print(msg, flush=True)

    stage = omni.usd.get_context().get_stage()

    # Reset + warmup
    env.reset()
    for _ in range(args.warmup_steps):
        env.step(torch.zeros((args.num_envs, 2), device=env.device))

    # Ensure Debug Xform
    if not stage.GetPrimAtPath("/World/envs/env_0/Debug").IsValid():
        stage.DefinePrim("/World/envs/env_0/Debug", "Xform")

    def get_sphere_positions():
        female_pos = get_world_position_usd(stage, args.female_sphere_prim)
        male_pos = get_world_position_usd(stage, args.male_sphere_prim)
        return female_pos, male_pos

    def compute_surface_xy(f, m):
        if f is None or m is None:
            return float('inf')
        dx, dy = f[0] - m[0], f[1] - m[1]
        return max(0.0, math.sqrt(dx*dx + dy*dy) - 0.01)

    def update_debug_spheres(female_pos, male_pos):
        if female_pos:
            f_vis = (female_pos[0], female_pos[1], female_pos[2] + args.debug_lift)
            create_debug_sphere(stage, "/World/envs/env_0/Debug/FemaleSphereVIS", f_vis, args.debug_radius, (1, 0, 0))
        if male_pos:
            m_vis = (male_pos[0], male_pos[1], male_pos[2] + args.debug_lift)
            create_debug_sphere(stage, "/World/envs/env_0/Debug/MaleSphereVIS", m_vis, args.debug_radius, (0, 0, 1))

    # Initial positions
    female_pos, male_pos = get_sphere_positions()
    log(f"[INITIAL] Female (SphereRear): {female_pos}")
    log(f"[INITIAL] Male (SpherePin):    {male_pos}")
    log(f"[INITIAL] Surface XY: {compute_surface_xy(female_pos, male_pos):.6f}")

    # Teleport
    if args.teleport == 1 and female_pos and male_pos:
        for k in range(3):
            female_pos, male_pos = get_sphere_positions()
            if female_pos is None or male_pos is None:
                log("[ERROR] Cannot read sphere positions")
                break

            sxy = compute_surface_xy(female_pos, male_pos)
            log(f"[TELEPORT iter={k}] surface_xy={sxy:.6f}")

            dx = male_pos[0] - female_pos[0]
            dy = male_pos[1] - female_pos[1]
            dz = male_pos[2] - female_pos[2]

            # Get robot root position
            root_pos = get_world_position_usd(stage, args.active_robot_root_path)
            if root_pos is None:
                log("[ERROR] Cannot read robot root")
                break

            # Set new position via USD
            new_pos = (root_pos[0] + dx, root_pos[1] + dy, root_pos[2] + dz)
            robot_prim = stage.GetPrimAtPath(args.active_robot_root_path)
            xf = UsdGeom.Xformable(robot_prim)
            
            # Find or create translate op
            translate_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xf.AddTranslateOp()
            
            translate_op.Set(Gf.Vec3d(new_pos[0], new_pos[1], new_pos[2]))
            log(f"[TELEPORT] delta=({dx:.4f},{dy:.4f},{dz:.4f})")

            # Settle
            for _ in range(args.settle_steps):
                env.step(torch.zeros((args.num_envs, 2), device=env.device))

        for _ in range(6):
            env.step(torch.zeros((args.num_envs, 2), device=env.device))

        female_pos, male_pos = get_sphere_positions()
        sxy = compute_surface_xy(female_pos, male_pos)
        log(f"[FINAL] surface_xy={sxy:.6f} (success if < 0.03)")

    # Update debug spheres
    female_pos, male_pos = get_sphere_positions()
    update_debug_spheres(female_pos, male_pos)

    # Compare with env internal
    env_female, env_male, env_sxy, _ = env.get_sphere_distances_from_physics()
    log(f"[ENV INTERNAL] female_pos: {env_female[0].cpu().numpy()}")
    log(f"[ENV INTERNAL] male_pos:   {env_male[0].cpu().numpy()}")
    log(f"[ENV INTERNAL] surface_xy: {float(env_sxy[0]):.6f}")
    log(f"[ACTUAL PRIMS] female_pos: {female_pos}")
    log(f"[ACTUAL PRIMS] male_pos:   {male_pos}")

    # Camera - center between both robots
    if female_pos and male_pos:
        center_x = (female_pos[0] + male_pos[0]) / 2
        center_y = (female_pos[1] + male_pos[1]) / 2
        look_at = (center_x, center_y, args.lookat_z)
    else:
        look_at = (0.5, 0.0, args.lookat_z)

    dx_cam, dy_cam = rotate_xy(args.cam_back, args.cam_side, args.cam_yaw_deg)
    cam_pos = (look_at[0] + dx_cam, look_at[1] + dy_cam, look_at[2] + args.cam_up)
    log(f"[CAMERA] pos={cam_pos} look_at={look_at}")

    with rep.new_layer():
        cam = rep.create.camera(position=cam_pos, look_at=look_at)
        rp = rep.create.render_product(cam, (args.stage_res_w, args.stage_res_h))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(stage_out), rgb=True)
    writer.attach([rp])
    rep.orchestrator.step()
    if hasattr(rep.orchestrator, "wait_until_complete"):
        rep.orchestrator.wait_until_complete()
    writer.detach()

    log(f"[SAVED] {stage_out}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
