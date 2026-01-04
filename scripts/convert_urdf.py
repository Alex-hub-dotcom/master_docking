#!/usr/bin/env python3
"""Converter URDF para USD"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(args)

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.converters.urdf_converter_cfg import JointDriveCfg

cfg = UrdfConverterCfg(
    asset_path='/workspace/teko/documents/CAD/Other_Formats/teko/teko.urdf',
    usd_dir='/workspace/teko/documents/CAD/USD/',
    usd_file_name='teko_fixed.usd',
    fix_base=False,
    make_instanceable=False,
    joint_drive=JointDriveCfg(
        gains=JointDriveCfg.GainsCfg(stiffness=0.0, damping=0.5)
    ),
)

print("Converting URDF to USD...")
converter = UrdfConverter(cfg)
print("Done! Saved to /workspace/teko/documents/CAD/USD/teko_fixed.usd")

app.app.close()
