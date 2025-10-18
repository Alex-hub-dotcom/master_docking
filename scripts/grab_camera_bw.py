# /workspace/teko/scripts/grab_camera_bw.py
from isaaclab.app import AppLauncher

def main(headless=False):
    app = AppLauncher(headless=headless).app

    import os, time
    from pathlib import Path
    import numpy as np
    import omni.replicator.core as rep


    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # camera USD que criamos no env
    CAM_PATH = "/World/envs/env_0/Robot/v1_teko_for_usd_ready/teko_body/teko_camera/cam_rpi_v2"

    # pasta de saída
    OUT_DIR = Path("/workspace/teko/_cam_out")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- sobe 1 env
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    env = TekoEnv(cfg)
    env.reset()

    # --- cria render product + writer
    with rep.new_layer():
        rp = rep.create.render_product(CAM_PATH, resolution=[1280, 960])
        writer = rep.WriterRegistry.get("BasicWriter")
        # salva RGB como png
        writer.initialize(
            output_dir=str(OUT_DIR),
            rgb=True,  # ativa RGB
            semantic_segmentation=False,
            instance_segmentation=False,
            bounding_box_2d_tight=False,
        )
        writer.attach([rp])

    # aquece sim e renderer um pouco
    for _ in range(30):
        env.step(env.actions)
        app.update()

    # roda alguns frames para o writer escrever
    num_frames = 10
    for i in range(num_frames):
        env.step(env.actions)
        app.update()
        rep.orchestrator.step()
        time.sleep(0.01)

    # procura um arquivo RGB salvo
    pngs = sorted(OUT_DIR.glob("rgb_*.png"))
    if not pngs:
        raise RuntimeError(f"Nenhum PNG salvo em {OUT_DIR}.")
    first = pngs[0]
    print(f"[grab] RGB salvo: {first}")

    # carrega e exporta em P&B (luma)
    try:
        import imageio.v2 as imageio
        rgb = imageio.imread(first)
        # garante HxWx3
        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[2] >= 3:
            gray = (0.299 * rgb[...,0] + 0.587 * rgb[...,1] + 0.114 * rgb[...,2]).astype(np.uint8)
        else:
            raise RuntimeError(f"Formato inesperado para RGB: {rgb.shape}")
        bw_path = OUT_DIR / "cam_bw.png"
        imageio.imwrite(bw_path.as_posix(), gray)
        print(f"[grab] GRAY salvo: {bw_path}  shape={gray.shape}")
    except Exception as e:
        # fallback para NPY se der ruim no PNG
        bw_npy = OUT_DIR / "cam_bw.npy"
        np.save(bw_npy.as_posix(), gray)
        print(f"[grab] falha no PNG ({e}). Salvei NPY: {bw_npy}")

    # encerra
    try:
        writer.detach([rp])
    except Exception:
        pass
    app.close()

if __name__ == "__main__":
    main(headless=False)
