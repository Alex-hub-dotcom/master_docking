# SPDX-License-Identifier: BSD-3-Clause
"""Câmera USD simples acoplada ao robô TEKO (sem depender de CameraCfg)."""

from __future__ import annotations
from omni.usd import get_context
from pxr import UsdGeom, Gf, Sdf

def ensure_teko_camera(resolution: tuple[int, int] = (640, 480)) -> str:
    """
    Garante uma câmera USD em /World/Robot/TekoCameraMount/TekoCamera.
    Retorna o caminho prim da câmera.
    """
    stage = get_context().get_stage()
    if stage is None:
        raise RuntimeError("Stage USD ainda não está disponível.")

    mount_path = Sdf.Path("/World/Robot/TekoCameraMount")
    cam_path = Sdf.Path("/World/Robot/TekoCameraMount/TekoCamera")

    # 1) Mount (Xform)
    if not stage.GetPrimAtPath(mount_path).IsValid():
        mount_prim = stage.DefinePrim(mount_path, "Xform")
        UsdGeom.XformCommonAPI(mount_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.15))
    else:
        mount_prim = stage.GetPrimAtPath(mount_path)

    # 2) Camera prim
    if not stage.GetPrimAtPath(cam_path).IsValid():
        cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    else:
        cam_prim = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))

    # 3) Pose (olhando para trás, com pequeno offset)
    xapi = UsdGeom.XformCommonAPI(cam_prim.GetPrim())
    xapi.SetRotate(Gf.Vec3f(0.0, 180.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xapi.SetTranslate(Gf.Vec3d(-0.06, 0.0, 0.0))

    # 4) Intrínsecos básicos
    cam_prim.GetFocalLengthAttr().Set(3.04)
    cam_prim.GetHorizontalApertureAttr().Set(3.68)
    cam_prim.GetVerticalApertureAttr().Set(2.76)
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    cam_prim.GetFocusDistanceAttr().Set(1.0)
    cam_prim.GetFStopAttr().Set(2.0)

    # 5) Atributo customizado de resolução (tipo correto!)
    w, h = int(resolution[0]), int(resolution[1])
    res_attr = cam_prim.GetPrim().GetAttribute("teko:resolution")
    if not res_attr:
        res_attr = cam_prim.GetPrim().CreateAttribute("teko:resolution", Sdf.ValueTypeNames.Int2)
    res_attr.Set(Gf.Vec2i(w, h))

    print(f"📷 Câmera TEKO garantida em {cam_path} (res={w}x{h})")
    return str(cam_path)
