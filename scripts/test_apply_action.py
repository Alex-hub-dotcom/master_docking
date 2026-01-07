#!/usr/bin/env python3
"""
TESTE DE CONFIGURAÇÕES DO _apply_action()
==========================================
Testa várias formas de converter [v, w] em torques para encontrar
a configuração que faz o robô mover e girar corretamente.
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


def test_config(env, name, torque_fn, steps=200):
    """Testa uma configuração de torques."""
    device = env.device
    
    results = {}
    
    for test_name, v, w in [("v=+1,w=0", 1.0, 0.0), ("v=0,w=+1", 0.0, 1.0), ("v=-1,w=0", -1.0, 0.0)]:
        env.reset()
        for _ in range(30):
            env.step(torch.tensor([[0.0, 0.0]], device=device))
        
        pos_init = env.robot.data.root_pos_w[0].clone()
        yaw_init = get_yaw(env.robot.data.root_quat_w[0])
        
        # Aplicar torques usando a função fornecida
        for _ in range(steps):
            torques = torque_fn(v, w, env._max_wheel_torque, device)
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
        dy = float(pos_final[1] - pos_init[1])
        dyaw = np.rad2deg(yaw_final - yaw_init)
        
        results[test_name] = {"dx": dx, "dy": dy, "dyaw": dyaw}
    
    return results


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
    print("TESTE DE CONFIGURAÇÕES DO _apply_action()")
    print("=" * 80)
    print(f"DOF order no cfg: {env.cfg.dof_names}")
    print(f"DOF indices: {env.dof_idx}")
    print(f"Ordem física: BackLeft=0, BackRight=1, FrontLeft=2, FrontRight=3")
    print()
    print("O robô spawna a 180° (frente aponta -X)")
    print("Esperado para v=+1,w=0: dx << 0 (move para frente), dyaw ≈ 0")
    print("Esperado para v=0,w=+1: dx ≈ 0, |dyaw| >> 0 (gira no lugar)")
    print("=" * 80)
    
    # A ordem dos slots no código é [FL, FR, BL, BR] mas mapeia para físicos [2, 3, 0, 1]
    # Quando aplicamos torques com dof_idx, o Isaac Lab reordena automaticamente
    
    configs = {}
    
    # Config 1: Atual (bugada)
    def config_atual(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        base = torch.tensor([[left, right, left, right]], device=dev) * max_t
        pol = torch.tensor([[1.0, -1.0, 1.0, -1.0]], device=dev)
        return base * pol
    
    configs["1. Atual (pol=[1,-1,1,-1])"] = config_atual
    
    # Config 2: Sem polaridade
    def config_sem_pol(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        return torch.tensor([[left, right, left, right]], device=dev) * max_t
    
    configs["2. Sem polaridade"] = config_sem_pol
    
    # Config 3: Polaridade só no right (corrigir convenção)
    def config_pol_right(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        # Aplicar pol=-1 só nas rodas direitas ANTES do stack
        right_corrected = right * (-1)
        return torch.tensor([[left, right_corrected, left, right_corrected]], device=dev) * max_t
    
    configs["3. Pol=-1 só no right"] = config_pol_right
    
    # Config 4: Baseado no comportamento físico observado
    # Físico 0 (BL): torque+ = frente → não precisa inverter
    # Físico 1 (BR): torque+ = frente → precisa inverter para diferencial funcionar!
    # Físico 2 (FL): torque+ = ~0 → problema físico
    # Físico 3 (FR): torque+ = trás → não precisa inverter
    def config_fisica(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        # Ordem: [FL, FR, BL, BR]
        # FL (físico 2): quase não responde, vamos assumir que precisa pol=1
        # FR (físico 3): torque+ = trás, então para frente precisamos pol=-1
        # BL (físico 0): torque+ = frente, pol=1
        # BR (físico 1): torque+ = frente (deveria ser trás!), pol=-1 para forçar comportamento correto
        pol = torch.tensor([[1.0, -1.0, 1.0, -1.0]], device=dev)
        # Mas aplicar ANTES de calcular left/right!
        # Na verdade, o problema é que BL e BR respondem igual...
        # Vamos testar aplicar a diferença L/R e depois corrigir individualmente
        torques = torch.tensor([[left, right, left, right]], device=dev) * max_t
        # Não multiplicar por polarity aqui, já que cancela!
        return torques
    
    # Config 5: Swap left/right para rodas traseiras
    # Se BL e BR ambas respondem como "esquerda", talvez o modelo tenha
    # as rodas traseiras com eixos espelhados
    def config_swap_back(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        # [FL, FR, BL, BR] - swap BL e BR
        return torch.tensor([[left, right, right, left]], device=dev) * max_t
    
    configs["4. Swap BL/BR"] = config_swap_back
    
    # Config 6: Inverter só BR
    def config_inv_br(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        # [FL, FR, BL, BR] com BR invertido
        return torch.tensor([[left, right, left, -right]], device=dev) * max_t
    
    configs["5. Inverter só BR"] = config_inv_br
    
    # Config 7: Usar só rodas traseiras (ignorar frontais problemáticas)
    def config_back_only(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        # [FL, FR, BL, BR] - só BL e BR ativos
        return torch.tensor([[0.0, 0.0, left, right]], device=dev) * max_t
    
    configs["6. Só rodas traseiras"] = config_back_only
    
    # Config 8: Só traseiras com BR invertido
    def config_back_only_inv(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        return torch.tensor([[0.0, 0.0, left, -right]], device=dev) * max_t
    
    configs["7. Só traseiras, BR inv"] = config_back_only_inv
    
    # Config 9: Todas com mesma polaridade base, diferencial correto
    def config_all_same(v, w, max_t, dev):
        k = 3.0
        left = torch.clamp(torch.tensor(v + k*w), -1, 1)
        right = torch.clamp(torch.tensor(v - k*w), -1, 1)
        # Todas positivas para v, diferença só em w
        # Se torque+ = frente em todas, então:
        return torch.tensor([[left, -right, left, -right]], device=dev) * max_t
    
    configs["8. left, -right pattern"] = config_all_same
    
    # Executar testes
    print()
    for name, fn in configs.items():
        print(f"\n{'='*80}")
        print(f"TESTANDO: {name}")
        print("-" * 80)
        
        results = test_config(env, name, fn)
        
        for test, r in results.items():
            dx, dy, dyaw = r["dx"], r["dy"], r["dyaw"]
            
            # Avaliar
            if test == "v=+1,w=0":
                linear_ok = "✓" if dx < -0.3 else "✗"
                angular_ok = "✓" if abs(dyaw) < 20 else "✗"
            elif test == "v=0,w=+1":
                linear_ok = "✓" if abs(dx) < 0.3 and abs(dy) < 0.3 else "~"
                angular_ok = "✓" if abs(dyaw) > 20 else "✗"
            else:  # v=-1
                linear_ok = "✓" if dx > 0.3 else "✗"
                angular_ok = "✓" if abs(dyaw) < 20 else "✗"
            
            print(f"  {test}: dx={dx:+.3f}m {linear_ok}, dyaw={dyaw:+.1f}° {angular_ok}")
    
    print("\n" + "=" * 80)
    print("LEGENDA:")
    print("  v=+1,w=0: Esperado dx << 0 (frente), dyaw ≈ 0")
    print("  v=0,w=+1: Esperado dx ≈ 0, |dyaw| >> 20°")
    print("  v=-1,w=0: Esperado dx >> 0 (trás), dyaw ≈ 0")
    print("=" * 80)
    
    env.close()
    sim.close()


if __name__ == "__main__":
    main()