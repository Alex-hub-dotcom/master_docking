# SPDX-License-Identifier: BSD-3-Clause
"""
Fluid movement test — TEKO multi-env (auto Isaac Sim)
-----------------------------------------------------
Inicia o Isaac Sim runtime antes de importar TekoEnv.
Compatível com Isaac Lab 0.47.1.
"""

from isaaclab.app import AppLauncher
import time
import torch

# ==== Inicializa Isaac Sim antes dos imports de isaaclab/teko ====
app_launcher = AppLauncher(headless=False)
app = app_launcher.app  # inicia o runtime aqui!

# ==== Agora é seguro importar o ambiente TEKO ====
from teko.tasks.direct.teko.teko_env import TekoEnv
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

# ==== Parâmetros de movimento ====
LINEAR_SPEED = 0.5
ANGULAR_FACTOR = 0.7
DURATION = 30.0
STEP_HZ = 30


# ==== Principal ====
def main():
    cfg = TekoEnvCfg()
    env = TekoEnv(cfg)

    # Reset seguro (novo método do TekoEnv)
    obs = env.reset()
    print(f"[INFO] TEKO fluid motion test iniciado com {env.scene.num_envs} ambiente(s).")

    steps = int(DURATION * STEP_HZ)
    dt = 1.0 / STEP_HZ
    start_time = time.time()

    num_envs = env.scene.num_envs
    device = env.device

    # Cria tensor fixo de ações base
    base_action = torch.tensor([[LINEAR_SPEED, LINEAR_SPEED * ANGULAR_FACTOR]],
                               device=device).repeat(num_envs, 1)

    # Loop de simulação
    for step in range(steps):
        env.step(base_action)

        if step % (STEP_HZ * 2) == 0:
            print(f"Step {step:04d}/{steps} | Tempo: {(step * dt):.1f}s")

    total_time = time.time() - start_time
    print(f"[INFO] Finalizado {steps} steps em {total_time:.1f}s.")

    app.close()


# ==== Entry point ====
if __name__ == "__main__":
    main()
