#!/usr/bin/env python3
"""
INVESTIGAÇÃO: Por que v=-1 (ré) não funciona?
=============================================
Config 5 funciona bem para v=+1 mas não para v=-1.
Vamos descobrir porquê.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(args)
sim = app.app

import torch
import numpy as np
import sys
sys.path.insert(0, "/workspace/teko/source/teko")

from teko.tasks.direct.teko.teko_env_state import TekoEnvState
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg


def get_yaw(quat):
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    return float(torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def main():
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.enable_curriculum = False
    
    env = TekoEnvState(cfg=cfg)
    device = env.device
    
    env.reset()
    for _ in range(50):
        env.step(torch.tensor([[0.0, 0.0]], device=device))
    
    print("=" * 80)
    print("INVESTIGAÇÃO: v=-1 (RÉ)")
    print("=" * 80)
    
    # Config 5: [left, right, left, -right]
    # Para v=+1: left=+1, right=+1 → torques=[+1, +1, +1, -1] * 10 = [10, 10, 10, -10]
    # Para v=-1: left=-1, right=-1 → torques=[-1, -1, -1, +1] * 10 = [-10, -10, -10, +10]
    
    print("\n[1] ANÁLISE TEÓRICA DOS TORQUES")
    print("-" * 80)
    
    for v in [1.0, -1.0]:
        left = v
        right = v
        torques = [left * 10, right * 10, left * 10, -right * 10]
        print(f"v={v:+.0f}: left={left:+.0f}, right={right:+.0f}")
        print(f"       torques [FL, FR, BL, BR] = {torques}")
        print(f"       FL={torques[0]:+.0f}, FR={torques[1]:+.0f}, BL={torques[2]:+.0f}, BR={torques[3]:+.0f}")
    
    print("\nObservação: Para v=-1, BR recebe +10 enquanto outros recebem -10")
    print("Isso pode causar o robô a girar em vez de ir reto para trás!")
    
    # Teste: v=-1 com diferentes configs
    print("\n" + "=" * 80)
    print("[2] TESTE DE DIFERENTES CONFIGS PARA v=-1")
    print("-" * 80)
    
    configs = {
        "Config 5 atual: [L, R, L, -R]": lambda v, w: (v, v, v, -v),
        "Simétrico: [L, R, L, R]": lambda v, w: (v, v, v, v),
        "Inverter FR também: [L, -R, L, -R]": lambda v, w: (v, -v, v, -v),
        "Só traseiras [0, 0, L, -R]": lambda v, w: (0, 0, v, -v),
        "Só traseiras simétricas [0, 0, L, R]": lambda v, w: (0, 0, v, v),
    }
    
    for name, fn in configs.items():
        env.reset()
        for _ in range(30):
            env.step(torch.tensor([[0.0, 0.0]], device=device))
        
        pos_init = env.robot.data.root_pos_w[0].clone()
        yaw_init = get_yaw(env.robot.data.root_quat_w[0])
        
        v, w = -1.0, 0.0
        fl, fr, bl, br = fn(v, w)
        torques = torch.tensor([[fl, fr, bl, br]], device=device) * env._max_wheel_torque
        
        for _ in range(200):
            env.robot.set_joint_effort_target(
                torques,
                env_ids=torch.tensor([0], device=device),
                joint_ids=env.dof_idx
            )
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.cfg.sim.dt)
        
        pos_final = env.robot.data.root_pos_w[0]
        yaw_final = get_yaw(env.robot.data.root_quat_w[0])
        
        dx = float(pos_final[0] - pos_init[0])
        dyaw = np.rad2deg(yaw_final - yaw_init)
        
        dx_ok = "✓" if dx > 0.3 else "✗"
        dyaw_ok = "✓" if abs(dyaw) < 20 else "✗"
        
        print(f"{name}")
        print(f"  torques=[{fl*10:+.0f}, {fr*10:+.0f}, {bl*10:+.0f}, {br*10:+.0f}]")
        print(f"  dx={dx:+.3f}m {dx_ok}, dyaw={dyaw:+.1f}° {dyaw_ok}")
        print()
    
    # Teste individual de cada roda com torque NEGATIVO
    print("=" * 80)
    print("[3] COMPORTAMENTO DE CADA RODA COM TORQUE NEGATIVO")
    print("-" * 80)
    print("(Para ré funcionar, torque- deveria mover o robô para +X)")
    
    for slot in range(4):
        wheel_name = env.cfg.dof_names[slot]
        
        env.reset()
        for _ in range(30):
            env.step(torch.tensor([[0.0, 0.0]], device=device))
        
        pos_init = env.robot.data.root_pos_w[0].clone()
        yaw_init = get_yaw(env.robot.data.root_quat_w[0])
        
        torques = torch.zeros((1, 4), device=device)
        torques[0, slot] = -env._max_wheel_torque  # Torque NEGATIVO
        
        for _ in range(150):
            env.robot.set_joint_effort_target(
                torques,
                env_ids=torch.tensor([0], device=device),
                joint_ids=env.dof_idx
            )
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.cfg.sim.dt)
        
        pos_final = env.robot.data.root_pos_w[0]
        yaw_final = get_yaw(env.robot.data.root_quat_w[0])
        
        dx = float(pos_final[0] - pos_init[0])
        dyaw = np.rad2deg(yaw_final - yaw_init)
        
        direction = "TRÁS(+X)" if dx > 0.05 else "FRENTE(-X)" if dx < -0.05 else "~0"
        
        print(f"Slot {slot} ({wheel_name[-15:]}) torque=-10:")
        print(f"  dx={dx:+.4f}m ({direction}), dyaw={dyaw:+.1f}°")
    
    # Comparação simétrica
    print("\n" + "=" * 80)
    print("[4] COMPARAÇÃO v=+1 vs v=-1 (torques simétricos em todas)")
    print("-" * 80)
    
    for v in [1.0, -1.0]:
        env.reset()
        for _ in range(30):
            env.step(torch.tensor([[0.0, 0.0]], device=device))
        
        pos_init = env.robot.data.root_pos_w[0].clone()
        yaw_init = get_yaw(env.robot.data.root_quat_w[0])
        
        # Torques simétricos: todas as rodas recebem o mesmo
        torques = torch.tensor([[v, v, v, v]], device=device) * env._max_wheel_torque
        
        for _ in range(200):
            env.robot.set_joint_effort_target(
                torques,
                env_ids=torch.tensor([0], device=device),
                joint_ids=env.dof_idx
            )
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.cfg.sim.dt)
        
        pos_final = env.robot.data.root_pos_w[0]
        yaw_final = get_yaw(env.robot.data.root_quat_w[0])
        
        dx = float(pos_final[0] - pos_init[0])
        dyaw = np.rad2deg(yaw_final - yaw_init)
        
        print(f"v={v:+.0f} (todas={v*10:+.0f}): dx={dx:+.3f}m, dyaw={dyaw:+.1f}°")
    
    print("\nSe v=+1 dá -X (frente) e v=-1 NÃO dá +X (trás) simétrico,")
    print("então há assimetria física no modelo USD!")
    
    # Teste final: Config corrigida para ré
    print("\n" + "=" * 80)
    print("[5] BUSCA DA CONFIG CORRETA PARA RÉ")
    print("-" * 80)
    print("Testando variações para encontrar o que funciona para v=-1")
    
    best_dx = 0
    best_config = None
    
    variations = [
        ("[-L, -R, -L, -R]", lambda: (-1, -1, -1, -1)),
        ("[-L, -R, -L, +R]", lambda: (-1, -1, -1, +1)),
        ("[-L, +R, -L, -R]", lambda: (-1, +1, -1, -1)),
        ("[-L, +R, -L, +R]", lambda: (-1, +1, -1, +1)),
        ("[+L, -R, +L, -R]", lambda: (+1, -1, +1, -1)),
        ("[+L, +R, +L, -R]", lambda: (+1, +1, +1, -1)),
        ("[0, 0, -L, -R]", lambda: (0, 0, -1, -1)),
        ("[0, 0, -L, +R]", lambda: (0, 0, -1, +1)),
        ("[0, 0, +L, -R]", lambda: (0, 0, +1, -1)),
        ("[0, 0, +L, +R]", lambda: (0, 0, +1, +1)),
    ]
    
    for name, fn in variations:
        env.reset()
        for _ in range(30):
            env.step(torch.tensor([[0.0, 0.0]], device=device))
        
        pos_init = env.robot.data.root_pos_w[0].clone()
        yaw_init = get_yaw(env.robot.data.root_quat_w[0])
        
        fl, fr, bl, br = fn()
        torques = torch.tensor([[fl, fr, bl, br]], device=device) * env._max_wheel_torque
        
        for _ in range(200):
            env.robot.set_joint_effort_target(
                torques,
                env_ids=torch.tensor([0], device=device),
                joint_ids=env.dof_idx
            )
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.cfg.sim.dt)
        
        pos_final = env.robot.data.root_pos_w[0]
        yaw_final = get_yaw(env.robot.data.root_quat_w[0])
        
        dx = float(pos_final[0] - pos_init[0])
        dyaw = np.rad2deg(yaw_final - yaw_init)
        
        dx_ok = "✓" if dx > 0.3 else ""
        dyaw_ok = "✓" if abs(dyaw) < 20 else ""
        
        if dx > best_dx and abs(dyaw) < 30:
            best_dx = dx
            best_config = name
        
        print(f"{name}: dx={dx:+.3f}m {dx_ok}, dyaw={dyaw:+.1f}° {dyaw_ok}")
    
    print(f"\nMelhor para RÉ: {best_config} com dx={best_dx:+.3f}m")
    
    print("\n" + "=" * 80)
    print("CONCLUSÃO")
    print("=" * 80)
    
    env.close()
    sim.close()


if __name__ == "__main__":
    main()