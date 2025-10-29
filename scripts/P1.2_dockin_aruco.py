# SPDX-License-Identifier: BSD-3-Clause
"""
P1.2 – Deteção ArUco + movimento automático até o marcador.
"""

from isaaclab.app import AppLauncher
import time
import numpy as np
import torch
import cv2

# ==== Configurações gerais ====
HEADLESS = False
DESIRED_SECONDS = 30
STEPS = DESIRED_SECONDS * 120
PRINT_EVERY = 20

# ==== Parâmetros ArUco (ajustados para distância) ====
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

# ==== Calibração da câmara (640x480, focal 3.6mm) ====
MARKER_SIZE = 0.05  # metros
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))


# ==== Função util: converter tensor em imagem ====
def to_numpy_img(t):
    if t is None:
        raise RuntimeError("Frame veio None.")
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


# ==== Função principal ====
def main():
    app = AppLauncher(headless=HEADLESS).app
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg = TekoEnvCfg()
    env = TekoEnv(cfg=cfg)
    env.reset()

    camera_obj = getattr(env, "camera", None)
    if camera_obj is None:
        raise RuntimeError("Câmara não encontrada no ambiente.")
    getter = getattr(camera_obj, "get_rgb", None)
    if getter is None:
        raise RuntimeError("A câmara não tem método get_rgb().")

    print("[INFO] Iniciando loop com deteção e movimento ArUco...")
    t0 = time.time()
    device = getattr(env, "device", "cuda:0")

    for step in range(1, STEPS + 1):
        frame = getter()
        img = to_numpy_img(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        vel_l = vel_r = 0.0  # parado por padrão

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

                # Frame do robô (REP-103)
                x_robot = z_cam
                y_robot = -x_cam
                z_robot = -y_cam

                rot_mat, _ = cv2.Rodrigues(rvec)
                yaw_rad = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                yaw_deg = np.degrees(yaw_rad)

                # --- Controle simples ---
                vel_forward = 0.4
                vel_turn = 0.2

                if abs(y_robot) > 0.02:
                    if y_robot > 0:
                        vel_l = vel_turn
                        vel_r = -vel_turn
                    else:
                        vel_l = -vel_turn
                        vel_r = vel_turn
                elif x_robot > 0.12:
                    vel_l = vel_r = vel_forward
                else:
                    vel_l = vel_r = 0.0  # parou

                if step % PRINT_EVERY == 0:
                    print(f"[ARUCO DETETADO] ID={ids[i][0]}")
                    print(f" ↪ Pos (ROBOT): x={x_robot:.3f}, y={y_robot:.3f}, z={z_robot:.3f}")
                    print(f" ↪ Yaw: {yaw_deg:.1f}° | vL={vel_l:.2f}, vR={vel_r:.2f}")
        else:
            if step % PRINT_EVERY == 0:
                print("Nenhum ArUco detectado.")

        # Aplicar ação (inverso por câmera traseira)
        action = torch.tensor([[-vel_l, -vel_r]], device=device)
        env.step(action)

    print(f"[INFO] Finalizado após {STEPS} steps ({time.time()-t0:.1f}s)")
    app.close()


if __name__ == "__main__":
    main()


# GOOD DOCKING  