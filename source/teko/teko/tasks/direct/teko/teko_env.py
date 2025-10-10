# Copyright (c) 2022-2025
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from .teko_env_cfg import TekoEnvCfg


def define_markers() -> VisualizationMarkers:
    """Create two arrow markers: robot forward (cyan) and command heading (red)."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class TekoEnv(DirectRLEnv):
    """Direct RL env for a 4-wheel TEKO robot."""

    cfg: TekoEnvCfg

    # -------------- base setup --------------

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        device = self.device
        num_envs = self.cfg.scene.num_envs

        # Actions
        a_dim = 4 if self.cfg.independent_wheels else 2
        self.actions = torch.zeros((num_envs, a_dim), device=device)

        # Smoothing buffers and limits
        self._targets_prev = torch.zeros((num_envs, 4), device=device)
        self._max_wheel_speed = float(getattr(self.cfg, "max_wheel_speed", 12.0))
        self._max_wheel_accel = float(getattr(self.cfg, "max_wheel_accel", 30.0))
        self._env_dt = float(self.cfg.decimation) * float(self.cfg.sim.dt)

        # Heading helpers
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        self.commands = self._sample_planar_unit_vectors(num_envs, device=device)
        self.yaws = torch.zeros((num_envs, 1), device=device)

        # Visualization buffers (filled after robot exists)
        self.marker_locations = None
        self.marker_offset = None
        self.forward_marker_orientations = None
        self.command_marker_orientations = None

        # Will be filled lazily after PhysX views exist
        self.dof_idx = None
        self.wheel_signs = None  # (4,)

        self._update_command_yaws()

    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground: high friction, zero restitution
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5,
                    dynamic_friction=1.2,
                    restitution=0.0,
                )
            ),
        )

        # Clone envs and register robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Markers
        self.visualization_markers = define_markers()

        # Visualization buffers
        device = self.device
        num_envs = self.cfg.scene.num_envs
        self.marker_locations = torch.zeros((num_envs, 3), device=device)
        self.marker_offset = torch.zeros((num_envs, 3), device=device)
        self.marker_offset[:, -1] = 0.8  # lift arrows
        self.forward_marker_orientations = torch.zeros((num_envs, 4), device=device)
        self.command_marker_orientations = torch.zeros((num_envs, 4), device=device)

    # -------------- helpers --------------

    def _sample_planar_unit_vectors(self, n: int, device) -> torch.Tensor:
        v = torch.randn((n, 3), device=device)
        v[:, -1] = 0.0
        norm = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
        return v / norm

    def _update_command_yaws(self):
        ratio = self.commands[:, 1] / (self.commands[:, 0] + 1e-8)
        gzero = self.commands > 0
        lzero = self.commands < 0
        plus = (lzero[:, 0] & gzero[:, 1])
        minus = (lzero[:, 0] & lzero[:, 1])
        offsets = torch.pi * plus - torch.pi * minus
        self.yaws = torch.atan(ratio).reshape(-1, 1) + offsets.reshape(-1, 1)

    def _build_wheel_signs(self):
        """Enforce same sign per side: FL==RL and FR==RR."""
        signs = torch.ones((len(self.dof_idx),), device=self.device, dtype=torch.float32)
        try:
            axes = self.robot.data.joint_axis[:, :3]
            raw = torch.sign(axes[self.dof_idx, 0]).to(torch.float32)
            raw[raw == 0] = 1.0
            raw[2] = raw[0]  # RL matches FL
            raw[3] = raw[1]  # RR matches FR
            signs = raw
        except Exception:
            signs = torch.tensor([+1.0, -1.0, +1.0, -1.0], device=self.device)
        self.wheel_signs = signs

    def _lazy_init_articulation(self):
        """Resolve joints and optionally set drive gains once PhysX views exist."""
        if self.dof_idx is not None and self.wheel_signs is not None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

        # Name → index
        joint_names = list(self.robot.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        dof_idx_list = [name_to_idx[n] for n in self.cfg.dof_names]
        self.dof_idx = torch.tensor(dof_idx_list, dtype=torch.long, device=self.device)

        # Per-wheel signs
        self._build_wheel_signs()

        # Velocity-like drive (if available in this build)
        n = len(self.dof_idx)
        stiffness = torch.zeros(n, device=self.device)
        damping = torch.full((n,), float(self.cfg.drive_damping), device=self.device)
        max_force = torch.full((n,), float(self.cfg.drive_max_force), device=self.device)
        if hasattr(self.robot, "set_joint_drive_property"):
            self.robot.set_joint_drive_property(
                stiffness=stiffness, damping=damping, max_force=max_force, joint_ids=self.dof_idx
            )

        print(f"[dbg] dof order [FL, FR, RL, RR] -> {self.cfg.dof_names}")
        print(f"[dbg] dof indices -> {self.dof_idx.tolist()}")
        print(f"[dbg] wheel_signs -> {self.wheel_signs.tolist()}")

    def _visualize_markers(self):
        if self.marker_locations is None:
            return
        if getattr(self.robot, "root_physx_view", None) is None:
            return

        pos = (self.robot.data.root_pos_w + self.marker_offset)
        f_quat = self.robot.data.root_quat_w
        c_quat = math_utils.quat_from_angle_axis(self.yaws, self.up_dir)

        N = pos.shape[0]
        idx0 = torch.zeros(N, dtype=torch.int64).cpu()
        idx1 = torch.ones(N, dtype=torch.int64).cpu()
        pos_c = pos.detach().contiguous().cpu()
        f_c = f_quat.detach().contiguous().cpu()
        c_c = c_quat.detach().contiguous().cpu()

        self.visualization_markers.visualize(pos_c, f_c, marker_indices=idx0)
        self.visualization_markers.visualize(pos_c, c_c, marker_indices=idx1)

    # -------------- RL API --------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is None or actions.numel() == 0:
            a_dim = 4 if self.cfg.independent_wheels else 2
            actions = 0.3 * torch.ones((self.cfg.scene.num_envs, a_dim), device=self.device)
        self.actions = actions.clone()

        self._lazy_init_articulation()

        if getattr(self.robot, "root_physx_view", None) is not None:
            self._visualize_markers()

    def _apply_action(self) -> None:
        if self.dof_idx is None:
            return

        # Build raw targets
        if self.cfg.independent_wheels:
            assert self.actions.shape[1] == 4
            raw = float(self.cfg.action_scale) * self.actions  # (N,4)
            # signs + polarity
            targets = raw * self.wheel_signs.unsqueeze(0) * float(self.cfg.wheel_polarity)
        else:
            assert self.actions.shape[1] == 2
            scale = float(self.cfg.action_scale)
            pol = float(self.cfg.wheel_polarity)
            left = (scale * self.actions[:, 0]).unsqueeze(-1)   # (N,1)
            right = (scale * self.actions[:, 1]).unsqueeze(-1)  # (N,1)
            # DOF order is [FL, FR, RL, RR]; feed [R, L, R, L] to fix swapped tracks
            raw = torch.hstack([right, left, right, left])      # (N,4)
            targets = raw * self.wheel_signs.unsqueeze(0) * pol

        # Slew-rate limit and saturation
        max_delta = self._max_wheel_accel * self._env_dt
        delta = torch.clamp(targets - self._targets_prev, -max_delta, +max_delta)
        targets = self._targets_prev + delta
        targets = torch.clamp(targets, -self._max_wheel_speed, +self._max_wheel_speed)
        self._targets_prev = targets.detach()

        # Send to sim
        self.robot.set_joint_velocity_target(targets, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        dot = torch.sum(forwards * self.commands, dim=-1, keepdim=True)
        cross_z = torch.cross(forwards, self.commands, dim=-1)[:, -1].reshape(-1, 1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        obs = torch.hstack((dot, cross_z, forward_speed))
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.robot.data.FORWARD_VEC_B)
        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0].reshape(-1, 1)
        alignment = torch.sum(forwards * self.commands, dim=-1, keepdim=True)
        return forward_speed + alignment

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self._lazy_init_articulation()

        if (self.commands is None) or (self.commands.shape[0] != self.cfg.scene.num_envs):
            self.commands = self._sample_planar_unit_vectors(self.cfg.scene.num_envs, device=self.device)
        self.commands[env_ids] = self._sample_planar_unit_vectors(len(env_ids), device=self.device)
        self._update_command_yaws()

        # Place above ground and zero velocities
        if getattr(self.robot, "root_physx_view", None) is not None:
            default_root_state = self.robot.data.default_root_state[env_ids]
            default_root_state[:, :3] = self.scene.env_origins[env_ids]
            default_root_state[:, 2] += float(self.cfg.spawn_height)
            default_root_state[:, 7:10] = 0.0
            default_root_state[:, 10:13] = 0.0
            self.robot.write_root_state_to_sim(default_root_state, env_ids)

            self._visualize_markers()


#python ./scripts/skrl/train.py --task=Template-Teko-Direct-v0