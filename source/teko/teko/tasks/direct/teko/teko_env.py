# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment — Isaac Lab 0.47.1
-----------------------------------
Active TEKO robot (RL agent) + static RobotGoal with emissive ArUco marker.
"""

from __future__ import annotations
import numpy as np
import torch
from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdLux, Gf, UsdShade

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaacsim.sensors.camera import Camera

from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION


class TekoEnv(DirectRLEnv):
    """TEKO environment: 1 robô ativo + 1 alvo estático com marcador ArUco."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        self._cam_res = (cfg.camera.width, cfg.camera.height)
        self._max_wheel_speed = cfg.max_wheel_speed
        self.actions = None
        self.dof_idx = None
        self.camera = None
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Setup da cena
    # ------------------------------------------------------------------
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not initialized")

        # --- Arena ---
        try:
            arena_path = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
            arena_prim = stage.DefinePrim(Sdf.Path("/World/StageArena"), "Xform")
            arena_prim.GetReferences().AddReference(arena_path)
        except Exception:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

      # --- Luz ambiente neutra + sol direcional ---
        # Remove dome default azul
        if stage.GetPrimAtPath("/World/DomeLight"):
            stage.RemovePrim("/World/DomeLight")

        # Luz ambiente uniforme (sem textura HDRI)
        ambient = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/AmbientLight"))
        ambient.CreateIntensityAttr(4000.0)
        ambient.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 0.95))  # branco ligeiramente quente
        ambient.CreateTextureFileAttr("")  # garante sem HDRI

        # Luz direcional (simula o sol)
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        sun.CreateIntensityAttr(2000.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))  # tom levemente quente (realista)
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-50.0)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(30.0)

        print("[INFO] Luz ambiente neutra + sol direcional aplicados (sem HDRI).")


        # --- Robô ativo ---
        self.robot = Articulation(TEKO_CONFIGURATION.replace(prim_path="/World/Robot"))
        self.scene.articulations["robot"] = self.robot

        # Pose fixa (sem randomização)
        robot_prim = stage.GetPrimAtPath("/World/Robot")
        xf_robot = UsdGeom.Xformable(robot_prim)
        xf_robot.ClearXformOpOrder()
        xf_robot.AddTranslateOp().Set(Gf.Vec3d(-0.2, 0.0, 0.43))
        xf_robot.AddRotateZOp().Set(180.0)


        # --- Robô alvo estático ---
        TEKO_USD_PATH = "/workspace/teko/documents/CAD/USD/teko_goal.usd"
        ARUCO_IMG_PATH = "/workspace/teko/documents/Aruco/test_marker.png"
        ROBOT_GOAL_PATH = "/World/RobotGoal"

        robot_goal = stage.DefinePrim(ROBOT_GOAL_PATH, "Xform")
        robot_goal.GetReferences().AddReference(TEKO_USD_PATH)
        xf_goal = UsdGeom.Xformable(robot_goal)
        xf_goal.AddTranslateOp().Set(Gf.Vec3f(1.0, 0.0, 0.40))
        xf_goal.AddRotateZOp().Set(180.0)

        # --- Placa ArUco ---
        size = 0.05
        half = size * 0.5
        ARUCO_PRIM_PATH = f"{ROBOT_GOAL_PATH}/Aruco"

        mesh = UsdGeom.Mesh.Define(stage, ARUCO_PRIM_PATH)
        mesh.CreatePointsAttr([
            Gf.Vec3f(0.0, -half, -half),  # canto inferior esquerdo
            Gf.Vec3f(0.0,  half, -half),  # canto inferior direito
            Gf.Vec3f(0.0,  half,  half),  # canto superior direito
            Gf.Vec3f(0.0, -half,  half),  # canto superior esquerdo
        ])


        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.CreateDoubleSidedAttr(True)

        xf_aruco = UsdGeom.Xformable(mesh)
        xf_aruco.AddTranslateOp().Set(Gf.Vec3f(0.17, 0.0, -0.045))
       

        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set([
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0)
        ])

        # --- Material plano (sem sombras, sem iluminação direta forte) ---
        LOOKS_PATH = f"{ROBOT_GOAL_PATH}/Looks/ArucoMaterial"
        material = UsdShade.Material.Define(stage, LOOKS_PATH)

        # Textura UV (imagem do ArUco)
        tex = UsdShade.Shader.Define(stage, LOOKS_PATH + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ARUCO_IMG_PATH))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")

        # Ligação UV (coordenadas de textura)
        st_reader = UsdShade.Shader.Define(stage, LOOKS_PATH + "/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader_output)

        # Saída RGB da textura
        tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # Shader principal (superfície)
        shader = UsdShade.Shader.Define(stage, LOOKS_PATH + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)

        # Conecta cor difusa à textura e remove brilho emissivo
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)


        # Finaliza material
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

        print("[INFO] ArUco material difuso aplicado com sucesso (sem emissive).")



        # --- Finaliza cena ---
        self.scene.clone_environments(copy_from_source=False)

        # --- Câmera ---
        sim = SimulationContext.instance()
        cam_path = self.cfg.camera.prim_path
        cam_prim = sim.stage.GetPrimAtPath(cam_path)
        if not cam_prim.IsValid():
            raise RuntimeError(f"[ERROR] Camera prim not found at {cam_path}")
        print(f"[INFO] Using active robot camera at {cam_path}")

        self.camera = Camera(
            prim_path=cam_path,
            resolution=self._cam_res,
            frequency=self.cfg.camera.frequency_hz,
        )
        self.camera.initialize()

    # ------------------------------------------------------------------
    # Física / Observações / Ações
    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return
        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        self.dof_idx = torch.tensor(
            [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx],
            dtype=torch.long, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        if self.dof_idx is None:
            return
        left, right = self.actions[0]
        targets = torch.tensor([left, right, left, right],
                               device=self.device).unsqueeze(0) * self._max_wheel_speed
        polarity = torch.tensor(self.cfg.wheel_polarity,
                                device=self.device).unsqueeze(0)
        self.robot.set_joint_velocity_target(
            targets * polarity,
            env_ids=torch.tensor([0], device=self.device),
            joint_ids=self.dof_idx)

    def _get_observations(self):
        obs = {}
        try:
            rgba = self.camera.get_rgba()
            if isinstance(rgba, np.ndarray) and rgba.size > 0:
                rgb = (torch.from_numpy(rgba[..., :3])
                       .to(self.device)
                       .permute(2, 0, 1)
                       .unsqueeze(0)
                       .float() / 255.0)
                obs["rgb"] = rgb
            else:
                h, w = self._cam_res[1], self._cam_res[0]
                obs["rgb"] = torch.zeros((1, 3, h, w), device=self.device)
        except Exception:
            h, w = self._cam_res[1], self._cam_res[0]
            obs["rgb"] = torch.zeros((1, 3, h, w), device=self.device)
        return obs

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        done = torch.zeros(1, dtype=torch.bool, device=self.device)
        return done, done

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()


