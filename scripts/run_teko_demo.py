from isaaclab.app import AppLauncher
import time, torch, pprint

def main(headless=False):
    app = AppLauncher(headless=headless).app

    # imports só depois do Kit
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # ---------- CFG: preset forte (sem mexer no arquivo do env) ----------
    cfg = TekoEnvCfg()
    cfg.scene.num_envs     = 1
    cfg.independent_wheels = False

    # potência e limites (ajuste aqui se quiser, não precisa tocar no env)
    cfg.action_scale    = 8.0
    cfg.wheel_polarity  = -1.0   # mantenha se “frente” já está correta
    cfg.drive_damping   = 40.0
    cfg.drive_max_force = 30.0
    cfg.max_wheel_speed = 10.0
    cfg.max_wheel_accel = 25.0
    cfg.ff_torque       = 6.0
    cfg.spawn_height    = 0.02

    print("\n=== CFG ATIVO (override do script) ===")
    show = {
        "num_envs": cfg.scene.num_envs,
        "independent_wheels": cfg.independent_wheels,
        "action_scale": cfg.action_scale,
        "wheel_polarity": cfg.wheel_polarity,
        "drive_damping": cfg.drive_damping,
        "drive_max_force": cfg.drive_max_force,
        "max_wheel_speed": cfg.max_wheel_speed,
        "max_wheel_accel": cfg.max_wheel_accel,
        "spawn_height": cfg.spawn_height,
        "ff_torque": cfg.ff_torque,
        "dof_names": cfg.dof_names,
    }
    pprint.pprint(show)

    env = TekoEnv(cfg)
    env.reset()

    # --------- AÇÕES: curva mais marcada ---------
    fwd   = torch.tensor([0.70, 0.70], device=env.device)   # frente
    curve = torch.tensor([0.95, 0.10], device=env.device)   # curva direita (L>R) forte
    spin  = torch.tensor([0.80,-0.80], device=env.device)   # girar direita (L+, R-)

    actions = torch.zeros((cfg.scene.num_envs, 2), device=env.device)

    def log(tag):
        v_b = env.robot.data.root_com_lin_vel_b[0]  # vx,vy no corpo
        w_w = env.robot.data.root_ang_vel_w[0]      # wz no mundo
        print(f"{tag}  v_body=({float(v_b[0]):+.3f},{float(v_b[1]):+.3f})  yaw_rate={float(w_w[2]):+.3f} rad/s")

    def one_phase(name, a, seconds=3.5, log_dt=0.5):
        print(f"\n[{name}] actions = {a.tolist()}")
        t0 = time.time()
        t_print = 0.0
        while app.is_running():
            actions[:] = a
            env.step(actions)
            app.update()
            t = time.time() - t0
            if t >= t_print:
                log(f"t={t:4.1f}s")
                t_print += log_dt
            if t >= seconds:
                break

    # primeiro step para lazy-init + map de juntas
    env.step(actions); app.update()
    joint_names = list(env.robot.joint_names)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    dof_idx = [name_to_idx[n] for n in cfg.dof_names]
    print("\n=== JOINT MAP ===")
    print("joint_names:", joint_names)
    print("dof order [FL, FR, RL, RR] ->", cfg.dof_names)
    print("dof indices ->", dof_idx)
    print("(wheel_signs internos = [+1,+1,+1,+1])")
    print(f"(wheel_polarity = {cfg.wheel_polarity})")

    # assentar 1s
    print("\n[Assentar] 1.0s")
    t0 = time.time()
    while app.is_running() and (time.time() - t0 < 1.0):
        actions[:] = 0.0
        env.step(actions); app.update()

    # fases (curva com mais tempo para ver o arco)
    one_phase("FRENTE",                fwd,   seconds=4.0, log_dt=0.5)
    one_phase("CURVA_DIREITA (L>R)",   curve, seconds=5.0, log_dt=0.5)
    one_phase("GIRO_DIREITA (L+, R-)", spin,  seconds=3.5, log_dt=0.5)

    print("\n[Parar]")
    for _ in range(15):
        actions[:] = 0.0
        env.step(actions); app.update()

    app.close()

if __name__ == "__main__":
    main(headless=False)
