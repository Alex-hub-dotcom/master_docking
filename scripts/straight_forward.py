# /workspace/teko/scripts/straight_forward.py
from isaaclab.app import AppLauncher
import time, torch

SPEED = 1.2
DURATION = 50.0

def main(headless=False):
    app = AppLauncher(headless=headless).app
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    cfg = TekoEnvCfg()  # uses num_envs=8 from cfg
    env = TekoEnv(cfg)
    env.reset()

    N = cfg.scene.num_envs
    actions = torch.full((N, 2), SPEED, device=env.device)

    t0 = time.time()
    try:
        while app.is_running() and time.time() - t0 < DURATION:
            env.step(actions)
            app.update()
    finally:
        app.close()

if __name__ == "__main__":
    main(headless=False)
