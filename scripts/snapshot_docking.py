# SPDX-License-Identifier: BSD-3-Clause
"""
Snapshot Docking - Diagrama esquemático do docking
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(argparse.Namespace(headless=True, enable_cameras=True))
simulation_app = app_launcher.app

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv


def run_to_docking():
    """Roda até conseguir um docking e tira foto."""
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    
    env = TekoEnv(cfg=cfg, render_mode=None)
    env.reset()
    
    print("[INFO] Movendo robô para posição de docking...")
    
    # Ação de ré mais forte
    backward_action = torch.tensor([[-1.0, 0.0]], device=env.device).expand(args.num_envs, -1)
    
    success = False
    for step in range(300):
        obs, reward, term, trunc, info = env.step(backward_action)
        
        _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
        dist = surface_xy[0].item()
        
        if step % 30 == 0:
            print(f"  Step {step}: dist={dist:.4f}m")
        
        if dist < 0.03 or term[0].item():
            print(f"[SUCCESS] Docking! dist={dist:.4f}m")
            success = True
            break
    
    if not success:
        print(f"[INFO] Distância final: {dist:.4f}m")
    
    # Criar diagrama
    create_schematic(env, success)
    
    # Salvar imagem da câmera do robô
    save_robot_camera(env)
    
    env.close()
    simulation_app.close()


def create_schematic(env, docked: bool):
    """Cria diagrama 2D esquemático da posição dos robôs."""
    
    robot_pos = env.robot.data.root_pos_w[0].cpu().numpy()
    goal_pos = env.goal_positions[0].cpu().numpy()
    female_pos, male_pos, surface_xy, _ = env.get_sphere_distances_from_physics()
    female = female_pos[0].cpu().numpy()
    male = male_pos[0].cpu().numpy()
    
    origin = env.scene.env_origins[0].cpu().numpy()
    
    # Converter para coordenadas locais
    robot_local = robot_pos - origin
    goal_local = goal_pos - origin
    female_local = female - origin
    male_local = male - origin
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Arena
    hx, hy = env._arena_half_x, env._arena_half_y
    arena_rect = plt.Rectangle(
        (-hx, -hy), 2*hx, 2*hy,
        fill=False, edgecolor='gray', linewidth=2, linestyle='--'
    )
    ax.add_patch(arena_rect)
    
    # Robô ativo (verde) - yaw=180 então comprimento em X
    aL = env._active_body_length
    aW = env._active_body_width
    active_rect = plt.Rectangle(
        (robot_local[0] - aL/2, robot_local[1] - aW/2), aL, aW,
        fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2
    )
    ax.add_patch(active_rect)
    ax.text(robot_local[0], robot_local[1], 'A', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Robô estático (vermelho)
    sL = env._static_body_length
    sW = env._static_body_width
    static_rect = plt.Rectangle(
        (goal_local[0] - sL/2, goal_local[1] - sW/2), sL, sW,
        fill=True, facecolor='lightcoral', edgecolor='red', linewidth=2
    )
    ax.add_patch(static_rect)
    ax.text(goal_local[0], goal_local[1], 'S', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Conectores
    ax.plot(female_local[0], female_local[1], 'mo', markersize=20, label='Female (traseira A)', zorder=5)
    ax.plot(male_local[0], male_local[1], 'co', markersize=20, label='Male (frente S)', zorder=5)
    
    # Linha entre conectores
    ax.plot([female_local[0], male_local[0]], [female_local[1], male_local[1]], 'k-', linewidth=3, zorder=4)
    
    # Distância
    dist_cm = surface_xy[0].item() * 100
    mid_x = (female_local[0] + male_local[0]) / 2
    mid_y = (female_local[1] + male_local[1]) / 2
    
    color = 'green' if docked else 'red'
    ax.text(mid_x, mid_y + 0.08, f'{dist_cm:.1f} cm', 
            ha='center', fontsize=14, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Setas indicando frente dos robôs (ambos yaw=180, frente para -X)
    arrow_len = 0.15
    ax.annotate('', xy=(robot_local[0] - arrow_len - aL/2, robot_local[1]),
                xytext=(robot_local[0] - aL/2, robot_local[1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(goal_local[0] - arrow_len - sL/2, goal_local[1]),
                xytext=(goal_local[0] - sL/2, goal_local[1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.set_xlim(-hx - 0.5, hx + 0.5)
    ax.set_ylim(-hy - 0.5, hy + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    
    status = "DOCKED ✓" if docked else "NOT DOCKED"
    ax.set_title(f'Vista de Cima - {status}\n'
                 f'Robô A: ({robot_local[0]:.3f}, {robot_local[1]:.3f})\n'
                 f'Robô S: ({goal_local[0]:.3f}, {goal_local[1]:.3f})\n'
                 f'Female: ({female_local[0]:.3f}, {female_local[1]:.3f}) | '
                 f'Male: ({male_local[0]:.3f}, {male_local[1]:.3f})',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/workspace/teko/docking_schematic.png", dpi=150)
    plt.close()
    
    print(f"[OK] Diagrama salvo em /workspace/teko/docking_schematic.png")


def save_robot_camera(env):
    """Salva imagem da câmera do robô."""
    try:
        cam = env.cameras[0]
        cam.update(dt=0.0)
        rgb_data = cam.data.output.get("rgb")
        
        if rgb_data is not None and rgb_data.numel() > 0:
            img = rgb_data.squeeze().cpu().numpy()
            if img.ndim == 3 and img.shape[-1] >= 3:
                plt.figure(figsize=(8, 8))
                plt.imshow(img[..., :3].astype(np.uint8))
                plt.title("Vista da câmera traseira do robô A")
                plt.axis('off')
                plt.savefig("/workspace/teko/robot_camera.png", dpi=150)
                plt.close()
                print("[OK] Câmera do robô salva em /workspace/teko/robot_camera.png")
    except Exception as e:
        print(f"[WARN] Não conseguiu salvar câmera do robô: {e}")


if __name__ == "__main__":
    run_to_docking()