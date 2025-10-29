# SPDX-License-Identifier: BSD-3-Clause
"""
P1.2 – ArUco detection and automatic docking movement (ground truth version)
----------------------------------------------------------------------------

This script detects an ArUco marker through the robot's camera and drives
the robot automatically toward the marker. When the marker is lost at a
short distance, the robot performs a controlled 5 cm forward movement.

The script logs the final positions of both robots (active and passive)
and saves the results in a CSV file as ground truth for docking evaluation.
"""

from isaaclab.app import AppLauncher
import time
import numpy as np
import torch
import cv2
import os

# ==== General configuration ====
HEADLESS = False
DESIRED_SECONDS = 30
STEPS = DESIRED_SECONDS * 120
PRINT_EVERY = 20

# ==== ArUco parameters (optimized for distance) ====
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 30
aruco_params.adaptiveThreshWinSizeStep = 3
aruco_params.adaptiveThreshConstant = 2.0
aruco_params.minMarkerPerimeterRate = 0.002
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.minCornerDistanceRate = 0.01
aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.01
aruco_params.perspectiveRemovePixelPerCell = 6
aruco_params.errorCorrectionRate = 0.7
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ==== Camera calibration (640x480, focal 3.6 mm) ====
MARKER_SIZE = 0.05  # meters
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))


# ==== Utility function: tensor → NumPy image ====
def to_numpy_img(t):
    """Convert a tensor or array to a uint8 NumPy RGB image."""
    if t is None:
        raise RuntimeError("Camera frame returned None.")
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in [3, 4]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.max() <= 1.0 + 1e-6:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


# ==== Main function ====
def main():
    app = AppLauncher(headless=HEADLESS).app
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg = TekoEnvCfg()
    env = TekoEnv(cfg=cfg)
    env.reset()

    camera_obj = getattr(env, "camera", None)
    if camera_obj is None:
        raise RuntimeError("Camera not found in environment.")
    getter = getattr(camera_obj, "get_rgb", None)
    if getter is None:
        raise RuntimeError("Camera object has no 'get_rgb()' method.")

    print("[INFO] Starting ArUco detection and motion control loop...")
    t0 = time.time()
    device = getattr(env, "device", "cuda:0")

    # Auxiliary variables
    last_x_robot = None
    move_forward_steps = 0
    y_filtered = 0.0  # exponential smoothing on Y

    for step in range(1, STEPS + 1):
        frame = getter()
        img = to_numpy_img(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        vel_l = vel_r = 0.0

        if ids is not None and len(ids) > 0:
            rvecs, tvecs = [], []
            objp = np.array([
                [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
                [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
                [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
            ], dtype=np.float32)

            for c in corners:
                ok, rvec, tvec = cv2.solvePnP(objp, c[0], camera_matrix, dist_coeffs)
                if ok:
                    rvecs.append(rvec)
                    tvecs.append(tvec)

            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                x_cam, y_cam, z_cam = tvec.flatten()
                x_robot = z_cam
                y_robot = -x_cam
                z_robot = -y_cam
                last_x_robot = x_robot

                # Exponential smoothing on y for stability
                y_filtered = 0.85 * y_filtered + 0.15 * y_robot

                rot_mat, _ = cv2.Rodrigues(rvec)
                yaw_rad = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                yaw_deg = np.degrees(yaw_rad)

                # === Motion control ===
                vel_forward = 0.3
                deadband = 0.001  # acceptable Y error (m)
                Kp = 1.8          # proportional gain

                if abs(y_filtered) > deadband:
                    # Rotate until roughly centered (slow and precise)
                    turn_speed = 0.12
                    vel_l = turn_speed if y_filtered > 0 else -turn_speed
                    vel_r = -turn_speed if y_filtered > 0 else turn_speed
                elif x_robot > 0.12:
                    # Move forward when aligned
                    vel_l = vel_r = vel_forward
                else:
                    vel_l = vel_r = 0.0

                if step % PRINT_EVERY == 0:
                    print(f"[ARUCO DETECTED] ID={ids[i][0]}")
                    print(f" ↪ Pos (Robot): x={x_robot:.4f}, y={y_robot:.4f} (filtered={y_filtered:.4f})")
                    print(f" ↪ Yaw: {yaw_deg:.2f}° | vL={vel_l:.3f}, vR={vel_r:.3f}")

        else:
            if last_x_robot is not None and last_x_robot <= 0.253:
                move_forward_steps = 20  # approx. 5 cm forward
                print("[INFO] ArUco lost — performing 5 cm final advance.")
            elif step % PRINT_EVERY == 0:
                print("No ArUco marker detected.")

        # === Extra forward motion (5 cm) ===
        if move_forward_steps > 0:
            vel_l = vel_r = 0.3
            move_forward_steps -= 1

        # Apply wheel velocities (camera is rear-mounted)
        action = torch.tensor([[-vel_l, -vel_r]], device=device, dtype=torch.float32)
        env.step(action)

    # ==== Ground truth logging ====
    pos_robot = env.robot.data.root_state_w[0, :3].cpu().numpy()
    try:
        passive = env.passive_robot.data.root_state_w[0, :3].cpu().numpy()
    except Exception:
        passive = np.array([np.nan, np.nan, np.nan])

    dist = np.linalg.norm(pos_robot - passive)
    print(f"[RESULT] Docking complete | Robot={pos_robot} | Target={passive} | Distance={dist:.4f} m")

    os.makedirs("/workspace/teko/logs", exist_ok=True)
    np.savetxt(
        "/workspace/teko/logs/docking_groundtruth.csv",
        np.hstack((pos_robot, passive, dist)),
        delimiter=",",
        header="robot_x,robot_y,robot_z,target_x,target_y,target_z,distance_m",
        comments=""
    )

    print(f"[INFO] Finished after {STEPS} steps ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()


# GROUND TRUTH SLOW CODE 