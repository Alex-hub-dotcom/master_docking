# /workspace/teko/scripts/P1.1_docking_aruco_detect.py
# Fase 1.1 – Leitura da câmara + deteção ArUco (compatível com OpenCV 4.10+)

from isaaclab.app import AppLauncher
import os
import time
import numpy as np
import torch
import cv2

# ==== Configurações gerais ====
HEADLESS = False
STEPS = 500
PRINT_EVERY = 10

# ==== Parâmetros ArUco (OpenCV 4.10+) ====
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
# Ajuste manual (necessário pois o construtor vem "vazio")
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 33
aruco_params.minMarkerPerimeterRate = 0.01
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.minCornerDistanceRate = 0.005
aruco_params.errorCorrectionRate = 0.7

# Novo objeto detector
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

MARKER_SIZE = 0.05  # metros

# Calibração aproximada da câmara
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# ==== util: converter tensor em imagem numpy ====
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

# ==== principal ====
def main():
    app = AppLauncher(headless=HEADLESS).app
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg = TekoEnvCfg()
    env = TekoEnv(cfg=cfg)
    env.reset()

    # localizar câmara
    camera_obj = getattr(env, "camera", None)
    if camera_obj is None:
        raise RuntimeError("Câmara não encontrada no ambiente.")
    getter = getattr(camera_obj, "get_rgba", None) or getattr(camera_obj, "get_rgb", None)
    if getter is None:
        raise RuntimeError("A câmara não tem get_rgba() nem get_rgb().")

    print("[INFO] Iniciando loop com deteção ArUco...")
    t0 = time.time()
    out_dir = "/workspace/teko/outputs/aruco_frames"
    os.makedirs(out_dir, exist_ok=True)

    for step in range(1, STEPS + 1):
        # ação neutra (robô parado)
        num_actions = getattr(env, "num_actions", 2)
        device = getattr(env, "device", "cuda:0")
        action = torch.zeros((1, num_actions), device=device)
        env.step(action)

        frame = getter()
        img = to_numpy_img(frame)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        if step == 1:
            cv2.imwrite(os.path.join(out_dir, "aruco_debug_gray.png"), gray)

        # ---- deteção ----
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, camera_matrix, dist_coeffs)
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                if step % PRINT_EVERY == 0:
                    print(f"[DETECTADO] ID={ids[i][0]} | Pos={tvec.flatten()} | Rot={rvec.flatten()}")
        else:
            if step % PRINT_EVERY == 0:
                print("Nenhum ArUco detectado.")

        if step % PRINT_EVERY == 0:
            cv2.imwrite(os.path.join(out_dir, f"aruco_{step:05d}.png"), img)

    print(f"[INFO] Finalizado após {STEPS} steps ({time.time()-t0:.1f}s)")
    app.close()

if __name__ == "__main__":
    main()
