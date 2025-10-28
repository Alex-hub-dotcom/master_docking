# /workspace/teko/scripts/P1_docking_aruco.py
# Fase 1: apenas ler a câmera (sem OpenCV)

from isaaclab.app import AppLauncher
import os
import time
import numpy as np
import torch

# ==== Configurações =====
HEADLESS = False
STEPS = 300                 # quantos passos rodar
PRINT_EVERY = 10            # imprime stats a cada N passos
SAVE_FRAMES = False         # opcional: salvar PNGs (requer Pillow)
SAVE_DIR = "/workspace/teko/outputs/frames_cam"

# ===== util: tenta importar Pillow sem quebrar =====
_pil_ok = False
if SAVE_FRAMES:
    try:
        from PIL import Image
        _pil_ok = True
        os.makedirs(SAVE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[WARN] Pillow não disponível ({e}). Não vou salvar imagens.")
        SAVE_FRAMES = False

def to_numpy_img(t):
    """
    Converte o tensor/imagem da câmera para (H,W,3) uint8 RGB.
    Aceita formatos (C,H,W), (H,W,C), (1,3,H,W), (1,H,W,3), etc.
    """
    import numpy as np
    import torch

    if t is None:
        raise RuntimeError("Frame veio None.")

    # converte para numpy se for tensor
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.array(t)

    # remove batch se existir
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # casos comuns:
    # (3,H,W) → ok (canal primeiro)
    # (H,W,3) → ok (canal último)
    # (480,640) ou (480,640,1) → inválido
    if arr.ndim == 3:
        if arr.shape[0] in [3, 4]:         # (C,H,W)
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] in [3, 4]:       # (H,W,C)
            pass  # já está correto
        else:
            raise RuntimeError(f"Formato inesperado da imagem: {arr.shape}")
    else:
        raise RuntimeError(f"Dimensões inesperadas: {arr.shape}")

    # descarta canal alpha se existir
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # normaliza para [0,255]
    if arr.max() <= 1.0 + 1e-6:
        arr = (arr * 255.0).clip(0, 255)

    return arr.astype(np.uint8)

def main():
    # Lança app
    app = AppLauncher(headless=HEADLESS).app

    # Importa aqui para garantir que o app subiu
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg=TekoEnvCfg()
    env = TekoEnv(cfg=cfg)
    
    env.reset()

    # tenta localizar o objeto da câmera
    # adapte este acesso caso o teu wrapper seja diferente
    camera_obj = None
    # tentativas comuns:
    if hasattr(env, "robot") and hasattr(env.robot, "camera"):
        camera_obj = env.robot.camera
    elif hasattr(env, "camera"):
        camera_obj = env.camera
    elif hasattr(env, "sensors") and "camera" in env.sensors:
        camera_obj = env.sensors["camera"]

    if camera_obj is None:
        raise RuntimeError("Não encontrei o objeto da câmera no ambiente.")

    # identifica função de captura disponível
    if hasattr(camera_obj, "get_rgba"):
        getter = camera_obj.get_rgba
    elif hasattr(camera_obj, "get_rgb"):
        getter = camera_obj.get_rgb
    else:
        raise RuntimeError("A câmera não possui get_rgba() nem get_rgb().")

    print("[INFO] Iniciando loop de captura da câmera...")
    t0 = time.time()
    for step in range(1, STEPS + 1):
        num_actions = getattr(env, "num_actions", 2)
        device = getattr(env, "device", "cuda:0")

        action = torch.zeros((1, num_actions,), device=device)
        env.step(action)
        # captura frame
        frame = getter()
        try:
            img = to_numpy_img(frame)  # (H,W,3) uint8
        except Exception as e:
            print(f"[ERROR] Falha ao converter frame: {e}")
            continue

        if step % PRINT_EVERY == 0:
            # estatísticas simples (em float32 para segurança)
            f32 = img.astype(np.float32)
            print(
                f"Step {step:04d} | shape={img.shape} | "
                f"min={f32.min():.1f} max={f32.max():.1f} mean={f32.mean():.1f}"
            )

        if SAVE_FRAMES and _pil_ok and (step % PRINT_EVERY == 0):
            fname = os.path.join(SAVE_DIR, f"cam_{step:05d}.png")
            try:
                Image.fromarray(img).save(fname)
            except Exception as e:
                print(f"[WARN] Não consegui salvar {fname}: {e}")

        # opcional: limitar um pouco o loop para não saturar
        time.sleep(0.0)

    dt = time.time() - t0
    print(f"[INFO] Finalizado. {STEPS} steps em {dt:.2f}s (≈{STEPS/dt:.1f} FPS lidos).")

    app.close()

if __name__ == "__main__":
    main()
 

# CAMERA IS WORKING