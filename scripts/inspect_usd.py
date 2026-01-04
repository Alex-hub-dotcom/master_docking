#!/usr/bin/env python3
"""Inspecionar USD do TEKO"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(args)

from pxr import Usd, UsdGeom, UsdPhysics

stage = Usd.Stage.Open("/workspace/teko/documents/CAD/USD/teko.usd")

print("=" * 60)
print("INSPEÇÃO DO USD - TEKO")
print("=" * 60)

for prim in stage.Traverse():
    name = prim.GetName()
    if "Wheel" in name or "Joint" in name:
        print(f"\n{prim.GetPath()}")
        print(f"  Type: {prim.GetTypeName()}")
        
        for attr in prim.GetAttributes():
            val = attr.Get()
            if val is not None and "axis" in attr.GetName().lower():
                print(f"  {attr.GetName()}: {val}")
            if val is not None and "orient" in attr.GetName().lower():
                print(f"  {attr.GetName()}: {val}")

print("\n" + "=" * 60)
app.app.close()
