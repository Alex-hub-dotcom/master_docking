# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Static Agent — Using teko_goal.usd
---------------------------------------
 Uses pre-made static USD (no camera, no actuators)
 Keeps collisions and gravity (falls naturally)
 Optionally adds ArUco marker for docking
"""

from __future__ import annotations

import time
from pxr import UsdGeom, UsdShade, Sdf, Gf
from isaaclab.sim import SimulationContext


class TEKOStatic:
    def __init__(self, prim_path: str, aruco_path: str | None = None):
        """
        Parameters
        ----------
        prim_path : str
            Where to spawn the static TEKO in the stage (e.g. /World/envs/env_0/RobotGoal).
        aruco_path : str | None
            Path to the ArUco texture image.
            - If None: no marker is created.
            - If not None: an ArUco card is attached to the robot.
        """
        self.prim_path = prim_path
        self.aruco_path = aruco_path
        self.usd_path = "/workspace/teko/documents/CAD/USD/teko_goal.usd"

        # Wait for SimulationContext
        sim = SimulationContext.instance()
        while sim is None or sim.stage is None:
            print("[WAIT] Waiting for SimulationContext stage...")
            time.sleep(0.2)
            sim = SimulationContext.instance()
        self.stage = sim.stage

        # Compose static robot
        self._load_usd()

        # Only create marker if a path is provided
        if self.aruco_path is not None:
            self._create_aruco_marker()
            print("[INFO] ArUco marker added to static TEKO.")
        else:
            print("[INFO] Static TEKO spawned WITHOUT ArUco marker.")

        print(f"[INFO] Static TEKO (teko_goal.usd) ready at {self.prim_path}")

    # ------------------------------------------------------------------
    def _load_usd(self):
        """Load the static TEKO from teko_goal.usd."""
        goal_prim = self.stage.DefinePrim(self.prim_path, "Xform")
        goal_prim.GetReferences().AddReference(self.usd_path)

        xf = UsdGeom.Xformable(goal_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3f(1.5, 0.0, 0.4))
        xf.AddRotateZOp().Set(180.0)
        print(f"[INFO] Loaded static TEKO from: {self.usd_path}")

    # ------------------------------------------------------------------
    def _create_aruco_marker(self):
        """Attach an ArUco marker to the static TEKO goal robot."""
        size = 0.05
        half = size * 0.5
        goal_path = self.prim_path
        aruco_img_path = self.aruco_path

        mesh_path = f"{goal_path}/Aruco"
        mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
        mesh.CreatePointsAttr([
            Gf.Vec3f(0.0, -half, -half),
            Gf.Vec3f(0.0,  half, -half),
            Gf.Vec3f(0.0,  half,  half),
            Gf.Vec3f(0.0, -half,  half),
        ])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.CreateDoubleSidedAttr(True)

        xf_aruco = UsdGeom.Xformable(mesh)
        xf_aruco.AddTranslateOp().Set(Gf.Vec3f(0.17, 0.0, -0.045))
        # xf_aruco.AddRotateYOp().Set(180.0)  # face toward active robot (optional)

        # Add UVs
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set([
            Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)
        ])

        # Material setup
        looks_path = f"{goal_path}/Looks/ArucoMaterial"
        material = UsdShade.Material.Define(self.stage, looks_path)

        tex = UsdShade.Shader.Define(self.stage, looks_path + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(aruco_img_path)
        )
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")

        st_reader = UsdShade.Shader.Define(self.stage, looks_path + "/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )

        shader = UsdShade.Shader.Define(self.stage, looks_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )

        material.CreateSurfaceOutput().ConnectToSource(
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
        UsdShade.MaterialBindingAPI(mesh).Bind(material)
