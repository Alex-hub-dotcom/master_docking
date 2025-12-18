# SPDX-License-Identifier: BSD-3-Clause
"""
P1.7 – ArUco-based docking (stable, light-independent, planar success)
- Adds bright CameraLight
- Ignores all markers except TARGET_ID
- Adaptive gamma + CLAHE for low-light robustness
- 1 cm forward nudge if marker lost near goal
- Saves connector-tip ground truth (robust to USD hierarchy)
"""

from isaaclab.app import AppLauncher
import os, time, json
import numpy as np
import torch
import cv2
from datetime import datetime

# ==== General configuration ====
HEADLESS = False
STEPS = 45 * 120
PRINT_EVERY = 20
REAR_CAMERA = True
STOP_DIST = 0.12
TARGET_ID = 1

# Control parameters
KP_DIST = 1.2
KP_YAW = 6.0
WHEELBASE_HALF = 0.22
V_MAX = 0.9
W_MAX = 1.2
BIAS = 0.10

# Lateral smoothing / integral clamp
Y_SMOOTH = 0.3
YI_GAIN = 0.10
YI_CLAMP = 0.05

# Camera + ArUco
MARKER_SIZE = 0.05
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# Connector radii
R_FEMALE = 0.005
R_MALE = 0.005

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def to_numpy_img(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (3, 4):
        a = np.transpose(a, (1, 2, 0))
    if a.shape[2] == 4:
        a = a[:, :, :3]
    if a.max() <= 1.0 + 1e-6:
        a = (a * 255.0).clip(0, 255)
    return a.astype(np.uint8)


def quaternion_to_euler(q):
    w, x, y, z = q
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.degrees([roll, pitch, yaw])


def add_camera_light(stage):
    from pxr import UsdLux, Gf, UsdGeom
    for name in ["Light", "defaultLight", "CameraLight"]:
        prim = stage.GetPrimAtPath(f"/World/{name}")
        if prim.IsValid():
            stage.RemovePrim(f"/World/{name}")
    light = UsdLux.SphereLight.Define(stage, "/World/CameraLight")
    light.CreateIntensityAttr(25000.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    light.CreateRadiusAttr(0.5)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 1.5))
    print("[INFO] Added automatic CameraLight (25 000 lm).")


# -----------------------------------------------------------------------------
# Ground-truth helpers
# -----------------------------------------------------------------------------
def get_sphere_world_positions(env):
    """Get world positions using corrected physics offsets."""
    # Use same corrected offsets as training
    FEMALE_OFFSET = torch.tensor([-0.245, 0.0, -0.07], device=env.device)
    MALE_OFFSET = torch.tensor([0.22667, -0.00144, -0.08815], device=env.device)
    
    # Active robot
    active_pos = env.robot.data.root_pos_w[0]
    active_quat = env.robot.data.root_quat_w[0]
    
    # Rotation matrix (same as training)
    w, x, y, z = active_quat[0], active_quat[1], active_quat[2], active_quat[3]
    R = torch.zeros((3, 3), device=env.device)
    R[0, 0] = 1 - 2*(y*y + z*z)
    R[0, 1] = 2*(x*y - w*z)
    R[0, 2] = 2*(x*z + w*y)
    R[1, 0] = 2*(x*y + w*z)
    R[1, 1] = 1 - 2*(x*x + z*z)
    R[1, 2] = 2*(y*z - w*x)
    R[2, 0] = 2*(x*z - w*y)
    R[2, 1] = 2*(y*z + w*x)
    R[2, 2] = 1 - 2*(x*x + y*y)
    
    female_tip = (active_pos + R @ FEMALE_OFFSET).cpu().numpy()
    
    # Static robot
    goal_pos = env.goal_positions[0]
    male_tip = (goal_pos + MALE_OFFSET).cpu().numpy()
    
    return female_tip, male_tip


def save_ground_truth(env, dur, thr=0.03):
    stage = env.scene.stage  # Add this line!
    
    # DEBUG: Check sphere type
    from pxr import UsdPhysics
    female_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/teko_urdf/TEKO_Body/TEKO_ConnectorRear/SphereRear")
    print(f"[DEBUG] Has RigidBodyAPI: {UsdPhysics.RigidBodyAPI(female_prim)}")
    print(f"[DEBUG] Prim type: {female_prim.GetTypeName()}")
    
    # Get sphere world positions
    female_tip, male_tip = get_sphere_world_positions(env)

    diff = female_tip - male_tip
    dist3d = float(np.linalg.norm(diff))
    distxy = float(np.linalg.norm(diff[:2]))

    # Subtract connector radii
    surface3d = max(0.0, dist3d - (R_FEMALE + R_MALE))
    surfacexy = max(0.0, distxy - (R_FEMALE + R_MALE))

    # success: planar (XY) alignment within threshold
    success = surfacexy <= thr

    rs = env.robot.data.root_state_w[0].detach().cpu().numpy()
    pos, quat = rs[:3], rs[3:7]
    euler = quaternion_to_euler(quat)

    data = {
        "timestamp": datetime.now().isoformat(),
        "duration_s": float(dur),
        "success_threshold_m": float(thr),
        "success": bool(success),
        "connector_positions": {
            "female_tip": female_tip.tolist(),
            "male_tip": male_tip.tolist(),
        },
        "distance_m": {
            "center_3d": dist3d,
            "center_xy": distxy,
            "surface_3d": surface3d,
            "surface_xy": surfacexy,
        },
        "robot_pose": {
            "position": pos.tolist(),
            "euler_deg": euler.tolist(),
        },
    }

    os.makedirs("/workspace/teko/logs", exist_ok=True)
    path = f"/workspace/teko/logs/docking_gt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(
        f"[INFO] Saved ground truth: {path} | "
        f"surface_xy={surfacexy*100:.1f} cm | surface_3d={surface3d*100:.1f} cm | success={success}"
    )
    return data


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    app = AppLauncher(headless=HEADLESS).app
    from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg

    cfg = TekoEnvCfg()
    cfg.episode_length_s = 10_000
    env = TekoEnv(cfg)
    env.reset()
    add_camera_light(env.scene.stage)

    # disable IsaacLab's auto-reset
    def _never_dones():
        n = env.scene.cfg.num_envs
        d = env.device
        return (
            torch.zeros((n,), dtype=torch.bool, device=d),
            torch.zeros((n,), dtype=torch.bool, device=d),
        )
    env._get_dones = _never_dones

    cam = env.cameras[0].get_rgba
    dev = env.device
    t0 = time.time()

    y_filt, y_int = 0.0, 0.0
    last_x = None

    for step in range(1, STEPS + 1):
        img = to_numpy_img(cam())
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        mean = gray.mean()
        gamma = 1.35 if mean < 100 else 0.85 if mean > 180 else 1.0
        gray = ((gray / 255.0) ** gamma * 255).astype(np.uint8)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        corners, ids, _ = detector.detectMarkers(gray)
        vL = vR = 0.0

        if ids is not None and TARGET_ID in ids.flatten():
            idx = int(np.where(ids.flatten() == TARGET_ID)[0][0])
            c = corners[idx]
            objp = np.array(
                [[-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                 [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                 [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                 [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]],
                dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, c[0], camera_matrix, dist_coeffs)
            if ok:
                x_cam, y_cam, z_cam = tvec.flatten()
                x_r, y_r = z_cam, -x_cam

                # Lateral PI controller
                y_filt = (1 - Y_SMOOTH) * y_filt + Y_SMOOTH * y_r
                y_int = np.clip(y_int + YI_GAIN * y_filt, -YI_CLAMP, YI_CLAMP)
                lat = y_filt + y_int
                fwd = max(0.0, x_r - STOP_DIST)

                v = KP_DIST * fwd
                w = KP_YAW * np.tanh(2.5 * lat)
                v = float(np.clip(v, 0.0, V_MAX))
                w = float(np.clip(w, -W_MAX, W_MAX))
                if REAR_CAMERA:
                    v = -v

                vL_cmd = v - WHEELBASE_HALF * w
                vR_cmd = v + WHEELBASE_HALF * w
                m = max(1e-6, np.max(np.abs([vL_cmd, vR_cmd])))
                vL = float(np.clip(vL_cmd / m + np.sign(vL_cmd) * BIAS, -1.0, 1.0))
                vR = float(np.clip(vR_cmd / m + np.sign(vR_cmd) * BIAS, -1.0, 1.0))

                last_x = x_r

                if x_r <= STOP_DIST:
                    print(f"[INFO] Dock reached: x={x_r:.3f}")
                    break

                if step % PRINT_EVERY == 0:
                    R, _ = cv2.Rodrigues(rvec)
                    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                    print(f"[ARUCO] ID={TARGET_ID} | x={x_r:.3f} | y={y_r:.3f} | yaw={yaw_deg:.1f}° | vL={vL:.2f} vR={vR:.2f}")

        else:
            # marker lost near goal → gentle forward nudge
            if last_x is not None and last_x <= (STOP_DIST + 0.10):
                vL = vR = (-0.2 if REAR_CAMERA else 0.2)
                for _ in range(12):
                    env.step(torch.tensor([[vL, vR]], device=dev, dtype=torch.float32))
                    app.update()
                print("[INFO] Marker lost near goal — 1 cm advance.")
                break
            elif step % PRINT_EVERY == 0:
                print("[INFO] Searching for ArUco marker...")

        env.step(torch.tensor([[vL, vR]], device=dev, dtype=torch.float32))
        app.update()

    save_ground_truth(env, time.time() - t0)
    env.close()
    app.close()


if __name__ == "__main__":
    main()